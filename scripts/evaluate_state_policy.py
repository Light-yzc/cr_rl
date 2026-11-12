"""Fixed-deck paired evaluation of explicit-state and legacy frozen policies.

Every physical 0.4s decision advances v3 memory, including forced WAITs. No
tactical priors, timing calibration, or WAIT bias are added. Successful-play
cost metrics come from the existing native replay parity audit, not requests.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.evaluation.cohort import paired_match_specs
from cr_rl.models import ActorCritic, ActorPolicy, ModelConfig
from cr_rl.models.policy import spatial_features
from cr_rl.models.state_policy import StateActorCritic, state_model_from_checkpoint
from cr_rl.training.persistence import atomic_json, file_sha256


DECISION_TICKS = 8
DECISION_SECONDS = .4


def latency_summary(seconds, *, budget_seconds=DECISION_SECONDS):
    values = np.asarray(seconds, dtype=np.float64)
    if not len(values):
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None,
                "max_ms": None, "budget_ms": 1000*budget_seconds, "violations": 0}
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid inference latency sample")
    return {"count": len(values), "p50_ms": float(np.quantile(values,.5)*1000),
            "p95_ms": float(np.quantile(values,.95)*1000),
            "p99_ms": float(np.quantile(values,.99)*1000), "max_ms": float(values.max()*1000),
            "budget_ms": budget_seconds*1000, "violations": int((values > budget_seconds).sum())}


def check_policy_compatibility(a, b):
    # Temporal width, layer count and window size intentionally may differ.
    fields = ("header_dim", "entity_feature_dim", "action_count", "vocab_size",
              "header_fields", "entity_fields")
    for name in fields:
        left, right = getattr(a,name), getattr(b,name)
        if name.endswith("fields"):
            left, right = tuple(left), tuple(right)
        if left != right:
            raise ValueError(f"Frozen policy observation/action contract differs: {name}")


def _restore_buffers(model, device):
    grid=torch.arange(576,device=device)
    xy=torch.stack(((grid%18+.5)/18,(grid//18+.5)/32),-1)
    model.tile_grid_features=spatial_features(xy,model.cfg.entity_dim)
    if hasattr(model.frame,"flag_bits"):
        model.frame.flag_bits=torch.arange(16,device=device)


def load_frozen_policy(path, device, placement_decoder=None):
    # The parent file also contains Adam/other tensors. mmap and a meta factory
    # avoid building duplicate CPU parameters or loading the optimizer.
    if not hasattr(np,"_core"):
        sys.modules.setdefault("numpy._core",np.core)
        sys.modules.setdefault("numpy._core.multiarray",np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric",np.core.numeric)
    checkpoint=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    version=checkpoint.get("model_architecture_version")
    info={"architecture":version,"optimizer_steps":checkpoint.get("optimizer_steps"),
          "checkpoint_calibration":checkpoint.get("calibration"),
          "decision_ticks":checkpoint.get("decision_ticks",5 if version==2 else None),
          "priors_disabled":True,"calibration_applied":False,"wait_logit_bias":0.}
    if version==3:
        if checkpoint.get("decision_ticks")!=DECISION_TICKS:
            raise ValueError("State checkpoint was not trained on the required 8tick interval")
        model=state_model_from_checkpoint(checkpoint,device=device)
    elif version==2:
        cfg=ModelConfig(**checkpoint["config"])
        if hasattr(cfg,"without_holding_prior"):
            cfg=cfg.without_holding_prior()
        with torch.device("meta"):
            model=(ActorPolicy if checkpoint.get("policy_only") else ActorCritic)(cfg)
        model.to_empty(device=device)
        _restore_buffers(model,device)
        model.load_state_dict(checkpoint["model"],strict=True)
    else:
        raise ValueError("Only architecture-v2 and architecture-v3 checkpoints are supported")
    if placement_decoder is not None:
        if version != 3:
            raise ValueError("Learned placement decoder requires architecture-v3")
        from cr_rl.models.learned_placement import attach_learned_placement
        payload = torch.load(placement_decoder, map_location="cpu", weights_only=False, mmap=True)
        attach_learned_placement(model, payload, source_checkpoint_sha256=file_sha256(path))
        info["placement_decoder_sha256"] = file_sha256(placement_decoder)
        info["placement_definition"] = model.learned_placement.get_extra_state()
        del payload
    model.eval()
    if getattr(model,"holding_prior",None) is not None:
        raise ValueError("Evaluation must not use a tactical holding prior")
    model.holding_prior = None
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise FloatingPointError("Frozen policy has nonfinite parameters")
    info["parameters"]=sum(parameter.numel() for parameter in model.parameters())
    info["context_len"]=model.cfg.context_len if version==2 else None
    info["grid_migration_limitation"]=(
        "Legacy BC is evaluated on 0.4s states instead of its original 0.25s grid; "
        "its context covers a different physical duration. This is not an unchanged-distribution comparison."
        if version==2 and info["decision_ticks"]!=DECISION_TICKS else None)
    del checkpoint
    gc.collect()
    return model,info


class PolicyStream:
    """One policy, one episode, one seat; never share mutable recurrent memory."""
    def __init__(self, model, adapter, device, *, precision="fp16", sampling_seed=0):
        self.model=model
        self.adapter=adapter
        self.device=torch.device(device)
        self.precision=precision
        self.state=None
        self.history=deque(maxlen=model.cfg.context_len)
        self.decisions=0
        self.generator=torch.Generator(device=self.device)
        self.generator.manual_seed(sampling_seed)

    def reset(self):
        self.state=None
        self.history.clear()
        self.decisions=0

    def autocast(self):
        if self.precision=="fp16":
            if self.device.type!="cuda":
                raise ValueError("FP16 online evaluation requires CUDA; CPU mocks use fp32")
            return torch.autocast("cuda",dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def sample(self, raw, legal):
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        schema=self.adapter.schema
        # No early return for only-WAIT masks: all physical frames advance memory.
        if isinstance(self.model,StateActorCritic) or getattr(self.model,"architecture_version",None)==3:
            header,entities,mask=self.adapter(np.asarray(raw)[None])
            tensors=[torch.as_tensor(value,device=self.device) for value in (header,entities,mask)]
            hand=torch.as_tensor(np.array(raw[list(schema.header["hand_card"])],copy=True)[None],device=self.device)
            legal_tensor=torch.as_tensor(np.array(legal,copy=True)[None],device=self.device,dtype=torch.bool)
            with self.autocast():
                if self.state is None:
                    self.state=self.model.initial_state(1,device=self.device)
                logp,_,self.state=self.model.step(*tensors,legal_tensor,state=self.state,
                    hand_card_ids=hand,compute_value=False)
        else:
            self.history.append(np.array(raw,copy=True))
            length=self.model.cfg.context_len
            windows=np.zeros((1,length,schema.seat_ints),dtype=np.int32)
            windows[0,-len(self.history):]=np.stack(self.history)
            time_mask=np.zeros((1,length),dtype=bool)
            time_mask[0,-len(self.history):]=True
            header,entities,mask=self.adapter(windows)
            tensor=lambda value:torch.as_tensor(value,device=self.device)
            hands=tensor(np.array(raw[list(schema.header["hand_card"])],copy=True)[None])
            decks=tensor(np.array(raw[list(schema.header["deck"])],copy=True)[None])
            seen=tensor(np.array(raw[list(schema.header["enemy_seen"])],copy=True)[None])
            with self.autocast():
                logp,_=self.model(tensor(header),tensor(entities),tensor(mask),
                    tensor(np.array(legal,copy=True)[None]).bool(),tensor(time_mask),
                    hand_card_ids=hands,deck_card_ids=decks,enemy_seen_card_ids=seen,compute_value=False)
        if not torch.isfinite(logp).all():
            raise FloatingPointError("Nonfinite frozen-policy log probabilities")
        probabilities=logp.float().exp()
        action=int(torch.multinomial(probabilities,1,generator=self.generator).cpu().item())
        latency=time.perf_counter()-started
        if not bool(legal[action]):
            raise ValueError("Frozen policy sampled an illegal action")
        self.decisions+=1
        return action,latency


def new_decision_audit():
    return {"decisions":0,"wait_decisions":0,"forced_wait_decisions":0,
        "card_requests":0,"ability_requests":0,"requested_card_elixir_total":0.,
        "requested_cost_at_least5":0,"full_elixir_wait_legal_seconds":0.,
        "longest_full_elixir_wait_legal_seconds":0.,"illegal_submissions":0,
        "_full_wait_run":0}


def record_decision(audit, raw, legal, action, schema):
    audit["decisions"]+=1
    audit["wait_decisions"]+=int(action==0)
    audit["forced_wait_decisions"]+=int(not np.asarray(legal)[1:].any())
    audit["card_requests"]+=int(0<action<2305)
    audit["ability_requests"]+=int(action>=2305)
    if 0<action<2305:
        slot=(action-1)//576
        cost=float(raw[schema.header["hand_cost"][slot]])
        audit["requested_card_elixir_total"]+=cost
        audit["requested_cost_at_least5"]+=int(cost>=5)
    full_wait=(action==0 and raw[schema.header["elixir"][0]]>=100000
        and np.asarray(legal)[1:2305].any())
    audit["_full_wait_run"]=audit["_full_wait_run"]+1 if full_wait else 0
    if full_wait:
        audit["full_elixir_wait_legal_seconds"]+=DECISION_SECONDS
        audit["longest_full_elixir_wait_legal_seconds"]=max(
            audit["longest_full_elixir_wait_legal_seconds"],audit["_full_wait_run"]*DECISION_SECONDS)


def finish_decision_audit(audit):
    result={k:v for k,v in audit.items() if not k.startswith("_")}
    count=result["card_requests"]
    result["mean_requested_card_cost"]=result["requested_card_elixir_total"]/count if count else None
    result["requested_cost_at_least5_fraction"]=result["requested_cost_at_least5"]/count if count else None
    result["scope"]="Active pre-step decisions; request costs are not confirmed native payments"
    return result


def native_strategy_summary(audit):
    result={}
    for role in ("learner","opponent"):
        plays=sum(game[role]["successful_card_plays"] for game in audit["games"])
        paid=sum(game[role]["paid_card_elixir_total"] for game in audit["games"])
        high=sum(game[role]["core_cost_thresholds"]["5"]["count"] for game in audit["games"])
        if any(not game.get("native_outcome_parity") for game in audit["games"]):
            raise ValueError("Native replay audit failed outcome parity")
        result[role]={"successful_card_plays":plays,
            "mean_paid_elixir_per_successful_card":paid/plays if plays else None,
            "cost_at_least5_fraction":high/plays if plays else None,
            "refused_card_commands":sum(game[role]["refused_card_commands"] for game in audit["games"]),
            "successful_ability_commands":sum(game[role]["successful_ability_commands"] for game in audit["games"])}
    return result


def evaluate_games(model, opponent, infos, league, matchups, args, *, env_factory=None):
    if env_factory is None:
        # Lazy import keeps CPU/unit checks independent of JPype/JVM installation.
        from cr_rl.envs.crforge import CrForgeVector
        env_factory=CrForgeVector
    candidate_latency=[]; opponent_latency=[]; simulator_latency=[]
    simulator_metadata=[]
    for wave_start in range(0,len(matchups),args.envs):
        wave=matchups[wave_start:wave_start+args.envs]
        count=len(wave)
        with env_factory(count,count,independent=True,arena="Arena_L",decision_ticks=DECISION_TICKS) as env:
            env.set_autoreset(False)
            seeds=[row["seed"] for row in wave]
            deck0=[league["decks"][row["deck0"]] for row in wave]
            deck1=[league["decks"][row["deck1"]] for row in wave]
            forms0=[league.get("slots",{}).get(row["deck0"],[0]*8) for row in wave]
            forms1=[league.get("slots",{}).get(row["deck1"],[0]*8) for row in wave]
            env.reset(seeds,deck0,deck1,forms0,forms1)
            if hasattr(env,"metadata"):
                metadata_path=args.out/f"simulator_wave_{wave_start:03d}.json"
                atomic_json(metadata_path,{"decision_ticks":DECISION_TICKS,
                    "matchups":wave,"simulator":env.metadata()})
                simulator_metadata.append({"file":metadata_path.name,"sha256":file_sha256(metadata_path)})
            schema=env.vector.schema
            adapter=CrForgeObservationAdapter(schema,env.vector.vocabulary_size,entity_bucket=16)
            for cfg in (model.cfg,opponent.cfg):
                if cfg.header_dim!=schema.header_ints or cfg.entity_feature_dim!=schema.entity_ints:
                    raise ValueError("Live simulator schema differs from checkpoint")
                if tuple(cfg.header_fields) and tuple(cfg.header_fields)!=tuple(schema.header_fields):
                    raise ValueError("Live simulator header fields differ from checkpoint")
                if tuple(cfg.entity_fields) and tuple(cfg.entity_fields)!=tuple(schema.entity_fields):
                    raise ValueError("Live simulator entity fields differ from checkpoint")
                if cfg.vocab_size!=env.vector.vocabulary_size:
                    raise ValueError("Live simulator vocabulary differs from checkpoint")
            streams=[]; audits=[]
            for match in wave:
                pair_streams=[]
                for seat in (0,1):
                    role=0 if seat==match["learner_seat"] else 1
                    seed=args.sampling_seed+match["pair"]*4+match["learner_seat"]*2+role
                    pair_streams.append(PolicyStream(model if role==0 else opponent,adapter,args.device,
                        precision=args.precision,sampling_seed=seed))
                streams.append(pair_streams)
                audits.append([new_decision_audit(),new_decision_audit()])
            active=np.ones(count,dtype=bool)
            for step in range(args.max_decisions):
                actions=np.zeros((count,2),dtype=np.int32)
                for i in np.flatnonzero(active):
                    for seat in (0,1):
                        raw=env.observations[i,seat]
                        legal=env.masks[i,seat]
                        action,seconds=streams[i][seat].sample(raw,legal)
                        actions[i,seat]=action
                        (candidate_latency if seat==wave[i]["learner_seat"] else opponent_latency).append(seconds)
                        record_decision(audits[i][seat],raw,legal,action,schema)
                started=time.perf_counter(); env.step(actions)
                simulator_latency.append(time.perf_counter()-started)
                for i in np.flatnonzero(active):
                    info=env.vector.info(int(i))
                    for seat in (0,1):
                        audits[i][seat]["illegal_submissions"]+=int(info.get(f"illegal_{seat}",0))
                    if env.dones[i] or info.get("fault") or info.get("truncated"):
                        learner=wave[i]["learner_seat"]
                        illegal=sum(a["illegal_submissions"] for a in audits[i])
                        valid=not info.get("fault") and not info.get("truncated") and not illegal
                        wave[i].update(steps=step+1,valid=bool(valid),info=info,
                            reward=float(env.rewards[i,learner]),decision_ticks=DECISION_TICKS,
                            illegal=illegal,decision_audit={"learner":finish_decision_audit(audits[i][learner]),
                                "opponent":finish_decision_audit(audits[i][1-learner])})
                        path=args.out/f"match_{wave_start+i:03d}.json"
                        try:
                            path.write_text(env.vector.lanes[i].replay_json(0),encoding="utf-8")
                            wave[i]["replay_sha256"]=file_sha256(path)
                        except Exception as error:
                            wave[i].update(valid=False,reason="native_replay_export_failed",
                                replay_export_error=str(error))
                        active[i]=False
                        for stream in streams[i]:stream.reset()
                if not active.any():break
                if step%128==0:
                    print(json.dumps({"wave_start":wave_start,"decision":step+1,
                        "remaining":int(active.sum())}),flush=True)
            for i in np.flatnonzero(active):
                wave[i].update(valid=False,reason="decision_limit_unfinished_not_a_draw",
                    steps=args.max_decisions,decision_ticks=DECISION_TICKS,
                    decision_audit={"learner":finish_decision_audit(audits[i][wave[i]["learner_seat"]]),
                        "opponent":finish_decision_audit(audits[i][1-wave[i]["learner_seat"]])})
                for stream in streams[i]:stream.reset()
    return {"candidate_end_to_end":latency_summary(candidate_latency),
        "opponent_end_to_end":latency_summary(opponent_latency),
        "simulator_vector_step":latency_summary(simulator_latency),
        "simulator_metadata":simulator_metadata,
        "scope":"adapter/history construction, H2D, forward/state update, sampling and CPU action; "
                "CUDA synchronized, batch1 per independent stream; simulator measured separately"}


def main():
    code_sha256=file_sha256(Path(__file__))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--policy-placement-decoder",type=Path,
        help="Optional decoder-only candidate, strictly bound to the policy checkpoint SHA")
    parser.add_argument("--opponent",type=Path,required=True)
    parser.add_argument("--decks",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--pairs",type=int,default=4)
    parser.add_argument("--pair-start",type=int,default=0)
    # A single lane is useful for local replay preview: with autoreset disabled,
    # finished lanes must not be stepped again by the vector backend.
    parser.add_argument("--envs",type=int,choices=(1,2,4),default=2)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--precision",choices=("fp16","fp32"),default="fp16")
    parser.add_argument("--max-decisions",type=int,default=1500,
        help="Hard failure after 600 simulated seconds by default; unfinished games are invalid")
    parser.add_argument("--sampling-seed",type=int,default=20261110)
    parser.add_argument("--simulator-seed-base",type=int,default=1000000)
    parser.add_argument("--pairing-seed",type=int,default=20261103)
    parser.add_argument("--native-audit",action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    if args.max_decisions*DECISION_SECONDS<360:
        parser.error("Decision limit must cover at least five minutes plus one-minute margin")
    if args.out.exists() and any(args.out.iterdir()):parser.error("Use a fresh evaluation directory")
    args.out.mkdir(parents=True,exist_ok=True)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose a GPU runtime before evaluation")
    import yaml
    league=yaml.safe_load(args.decks.read_text(encoding="utf-8"))
    matchups=paired_match_specs(list(league["decks"]),args.pairs,args.pair_start,
        simulator_seed_base=args.simulator_seed_base,pairing_seed=args.pairing_seed)
    torch.manual_seed(args.sampling_seed)
    started=time.perf_counter()
    policy_sha256=file_sha256(args.policy)
    opponent_sha256=file_sha256(args.opponent)
    decoder_sha256=file_sha256(args.policy_placement_decoder) if args.policy_placement_decoder else None
    model,model_info=load_frozen_policy(args.policy,args.device,args.policy_placement_decoder)
    opponent,opponent_info=load_frozen_policy(args.opponent,args.device)
    check_policy_compatibility(model.cfg,opponent.cfg)
    latency=evaluate_games(model,opponent,[model_info,opponent_info],league,matchups,args)
    valid=[row for row in matchups if row.get("valid")]
    rewards=[row["reward"] for row in valid]
    if policy_sha256!=file_sha256(args.policy) or opponent_sha256!=file_sha256(args.opponent):
        raise ValueError("Checkpoint changed during frozen evaluation")
    if args.policy_placement_decoder and decoder_sha256 != file_sha256(args.policy_placement_decoder):
        raise ValueError("Learned placement decoder changed during frozen evaluation")
    if code_sha256!=file_sha256(Path(__file__)):
        raise ValueError("Evaluation implementation changed during frozen evaluation")
    result={"policy_sha256":policy_sha256,"opponent_sha256":opponent_sha256,
        "deck_pool_sha256":file_sha256(args.decks),"evaluation_code_sha256":code_sha256,
        "policy":model_info,"opponent":opponent_info,"decision_ticks":DECISION_TICKS,
        "pair_count":args.pairs,"pair_start":args.pair_start,"pairing_seed":args.pairing_seed,
        "simulator_seed_base":args.simulator_seed_base,"sampling_seed":args.sampling_seed,
        "sampling_protocol":"Independent torch generator per physical game/role; stable across wave sizes",
        "protocol":"fixed physical decks/seeds, paired policy-seat swap, stochastic sampling; explicit placement calibration is recorded in policy definition",
        "envs":args.envs,"precision":args.precision,"max_decisions":args.max_decisions,
        "games":len(valid),"invalid_games":len(matchups)-len(valid),
        "fault_games":sum(bool(row.get("info",{}).get("fault")) for row in matchups),
        "illegal_submissions":sum(row.get("illegal",0) for row in matchups),
        "wins":sum(r>0 for r in rewards),"losses":sum(r<0 for r in rewards),"draws":sum(r==0 for r in rewards),
        "matchups":matchups,"latency":latency,
        "candidate_inference_budget_passed":latency["candidate_end_to_end"]["violations"]==0,
        "seconds":time.perf_counter()-started,
        "limitation":"Small fixed paired evaluation; neither loss nor this score establishes convergence"}
    evaluation=args.out/"evaluation.json"
    atomic_json(evaluation,result)
    if len(valid)!=len(matchups):
        raise RuntimeError("Evaluation has unfinished, truncated, illegal or faulted matches")
    if not result["candidate_inference_budget_passed"]:
        raise RuntimeError("Candidate exceeded the required 400ms decision budget; evidence was saved")
    summary={"evaluation":str(evaluation),"evaluation_sha256":file_sha256(evaluation),
        "native_audited":False,"latency":latency,"games":len(valid),
        "wins":result["wins"],"losses":result["losses"],"draws":result["draws"]}
    if args.native_audit:
        audit_path=args.out/"execution_audit.json"
        command=[sys.executable,str(Path(__file__).with_name("audit_policy_replays.py")),
            "--evaluation",str(evaluation.resolve()),"--out",str(audit_path.resolve())]
        with (args.out/"native_audit.log").open("w",encoding="utf-8") as output:
            subprocess.run(command,cwd=Path(__file__).resolve().parents[1],
                stdout=output,stderr=subprocess.STDOUT,check=True)
        audit=json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["evaluation_sha256"]!=file_sha256(evaluation) or len(audit["games"])!=len(matchups):
            raise ValueError("Native audit belongs to different frozen evaluation evidence")
        summary.update(native_audited=True,native_strategy=native_strategy_summary(audit),
            execution_audit=str(audit_path),execution_audit_sha256=file_sha256(audit_path))
    else:
        summary["limitation"]="Smoke only: successful-play costs have not been audited natively"
    atomic_json(args.out/"evaluation_summary.json",summary)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
