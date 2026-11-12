"""Re-simulate original commands on an explicit causal decision grid.

Example: --metadata old.json --source-parquet IL_Replay_replays.parquet
         --source-root .tmp_latest --decision-ticks 8 --limit 8 --out new_run
         --simulator-install .cache/vendor/crforge/env/build/install/env
         --tables .cache/replay-viewer/tables/16.635.3

--plan-only does not import JVM/Torch or write a training manifest. Missing full
source commands are rejected unless --allow-recorded-subset is explicit; that
mode always marks the dataset as smoke_only and not training_ready.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4*1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_helpers(source_root):
    # Do not silently mix old root modules with the current live repair code.
    sys.path.insert(0, str(source_root/"src"))
    local = Path(__file__).resolve().parents[1]/"src/cr_rl/data/interval_rebuild.py"
    spec = importlib.util.spec_from_file_location("interval_rebuild", local)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse((source_root/"scripts/import_il_replay.py").read_text(encoding="utf-8"))
    aliases = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALIASES" for t in node.targets))
    return module, aliases


def parquet_events(path, metadata, selected):
    import pyarrow.parquet as pq
    expected = metadata.get("source_file_sha256")
    actual = digest(path)
    if expected and actual != expected:
        raise ValueError("Source parquet SHA256 differs from the original dataset")
    wanted = {int(metadata["replays"][i]["source_row_index"]): i for i in selected}
    if len(wanted) != len(selected):
        raise ValueError("Multiple replay records refer to the same original source row")
    found = {}
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["replay_tag", "payload_json"]):
        for index, row in enumerate(batch.to_pylist(), start=offset):
            if index not in wanted:
                continue
            replay_id = wanted[index]
            payload = json.loads(row["payload_json"])
            replay = metadata["replays"][replay_id]
            tag = payload.get("source", {}).get("replay_tag", row["replay_tag"])
            if tag != replay["replay_tag"]:
                raise ValueError("Source row replay identity differs from metadata")
            events = [event for event in payload.get("events", []) if event.get("kind") in
                {"play_card", "ability", "activate_ability", "ability_activation"}]
            if len(events) != replay["total_play_events"]:
                raise ValueError("Source command count differs from recorded original stream")
            found[replay_id] = (events, {"mode": "original_parquet", "source_complete": True,
                "source_command_coverage": 1., "declared_source_events": len(events),
                "available_source_events": len(events), "source_parquet_sha256": actual})
        offset += batch.num_rows
        if len(found) == len(wanted):
            break
    if len(found) != len(wanted):
        raise ValueError("Source parquet lacks requested original replay rows")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(".tmp_latest"))
    parser.add_argument("--simulator-install", type=Path)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decision-ticks", type=int, default=8)
    parser.add_argument("--limit", type=int, default=8, help="0 selects all source games")
    parser.add_argument("--start-replay", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-recorded-subset", action="store_true",
        help="Smoke only: permit missing source events; never mark training-ready")
    parser.add_argument("--skip-incomplete-ledgers", action="store_true",
        help="Without parquet, select only games with every original event in ledger")
    args = parser.parse_args()
    if args.decision_ticks < 1 or min(args.limit,args.start_replay) < 0:
        parser.error("Positive interval and nonnegative limits required")
    if not args.plan_only and not (args.simulator_install and args.tables):
        parser.error("Live rebuild requires explicit --simulator-install and --tables")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a fresh output directory; old data is never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    helpers, aliases = load_helpers(args.source_root.resolve())
    original = json.loads(args.metadata.read_text(encoding="utf-8"))
    selected = list(range(args.start_replay,len(original["replays"])))
    if args.skip_incomplete_ledgers and not args.source_parquet:
        usable=[]
        for replay_id in selected:
            try: helpers.recorded_source_events(original["replays"][replay_id])
            except ValueError: continue
            usable.append(replay_id)
        selected=usable
    if args.limit: selected=selected[:args.limit]
    source = parquet_events(args.source_parquet,original,selected) if args.source_parquet else {}
    plans=[]
    for replay_id in selected:
        replay=original["replays"][replay_id]
        events,provenance=source.get(replay_id,(None,None))
        if events is None:
            events,provenance=helpers.recorded_source_events(replay,
                allow_incomplete=args.allow_recorded_subset)
        canonical=helpers.canonicalise_events(events,replay["decks"],aliases)
        plan,summary=helpers.causal_grid_plan(canonical,args.decision_ticks)
        plans.append((replay_id,canonical,provenance,summary))
    if not plans: raise ValueError("No source command streams selected")
    protocol={"version":"crrl-causal-interval-rebuild-v2-strict", "decision_ticks":args.decision_ticks,
        "tick_ms":50,"source_metadata_sha256":digest(args.metadata),"source_root":str(args.source_root),
        "rebuild_script_sha256":digest(Path(__file__).resolve()),
        "interval_helper_sha256":digest(Path(__file__).resolve().parents[1]/"src/cr_rl/data/interval_rebuild.py"),
        "live_executor_sha256":digest(args.source_root/"src/cr_rl/data/replay_execution.py"),
        "source_parquet":str(args.source_parquet) if args.source_parquet else None,
        "allow_recorded_subset":args.allow_recorded_subset,"smoke_only":args.allow_recorded_subset,
        "label_timing":"causal ceil grid then FIFO/native-mask waits; never advance source commands",
        "admission":"every original command maps one-to-one to unique native-confirmed flags0 legal row",
        "source_events_retained_in_external_audit":True,
        "serial_native_confirmation":False,"skip_unavailable_abilities":False,
        "not_downsampling":True,"source_replay_ids":selected,
        "plans":[{"original_replay_id":i,"source":p,"grid":s} for i,_,p,s in plans]}
    atomic_json(args.out/"interval_protocol.json",protocol)
    if args.plan_only:
        print(json.dumps({"phase":"planned","games":len(plans),"training_ready":False,
            "source_events":sum(len(e) for _,e,_,_ in plans),"out":str(args.out)}))
        return

    import numpy as np
    from crforge_env import LadderConfig,VectorBattle
    from cr_rl.data.native_replay import NativeReplayObserver
    from cr_rl.data.opening_cycle import native_opening_deals,reorder_for_deal
    from cr_rl.data.replay_execution import ExecutionConfig,execute_repaired_events
    from cr_rl.data.replay_supervision import mask_omitted_commands
    from cr_rl.envs.backend_patch import installed_backend,loaded_backend

    provenance=installed_backend(args.simulator_install,require_patch=True)
    expected=original.get("backend_patch_sha256")
    # This is a fresh simulation, not a reproduction of the old repaired array.
    # installed_backend verifies the *current* registered patched build. Record
    # its difference from the source provenance; requiring the old hash here
    # would reject a deliberate rebuild with the current late-champion fix.
    target_patch=provenance.get("patch_manifest_sha256")
    protocol.update(source_backend_patch_sha256=expected,
        target_backend_patch_sha256=target_patch,backend_changed=target_patch!=expected)
    atomic_json(args.out/"interval_protocol.json",protocol)
    policy=original["import_policy"]
    config=LadderConfig(decision_ticks=args.decision_ticks,level=int(original["level"]),
        tower_level=int(original["tower_level"]),mask_version=2,arena="Arena_L")
    shards=[]; new_replays=[]; results=[]; total_rows=0
    (args.out/"shards").mkdir();(args.out/"audit").mkdir()
    with VectorBattle(1,1,install=args.simulator_install,tables=args.tables,config=config) as vector:
        vector.set_autoreset(False)
        loaded_backend(args.simulator_install)
        vector_metadata=dict(vector.metadata)
        for original_id,events,source_provenance,grid_summary in plans:
            replay=original["replays"][original_id]
            decks=replay["decks"]; slots=replay["slots"]
            seed=int(replay["initial_simulator_seed"])
            vector.reset([seed],decks[0],decks[1],slots[0],slots[1])
            runtime=[list(deck) for deck in decks]; runtime_slots=[list(s) for s in slots]
            native=native_opening_deals(vector)
            for seat,inferred in enumerate(replay.get("opening_cycle_inference",[])):
                if inferred:
                    runtime[seat],runtime_slots[seat]=reorder_for_deal(decks[seat],slots[seat],
                        native[seat]["order"],inferred["order"])
            vector.reset([seed],runtime[0],runtime[1],runtime_slots[0],runtime_slots[1])
            event_map={}
            for event in events: event_map.setdefault(int(event["replay_tick_20hz"]),[]).append(event)
            stats,arrays,_,_=execute_repaired_events(vector,event_map,decks,decks,runtime,
                ExecutionConfig(decision_ticks=args.decision_ticks,repair_hand=policy.get("repair_hand",False),
                    max_hand_repairs=policy.get("max_hand_repairs",-1),repair_tile=policy.get("repair_tile",False),
                    max_tile_shift=policy.get("max_tile_shift",2),repair_time=True,
                    time_tolerance_ticks=args.decision_ticks-1,max_delay_ticks=policy.get("max_delay_ticks",200),
                    max_omissions=0,skip_unavailable_abilities=False,serial_native_confirmation=False,capture_rows=True,
                    source_end_tick=replay.get("source_declared_duration_ticks")),
                native_observer=NativeReplayObserver(vector))
            audit=helpers.execution_interval_audit(stats,len(events))
            if arrays is not None:
                arrays["flags"]=mask_omitted_commands(arrays["flags"],arrays["ticks"],arrays["seats"],
                    stats["omitted_command_ticks"],decision_ticks=args.decision_ticks)
            admission=helpers.strict_command_admission(stats,arrays,events,args.decision_ticks)
            success=admission["accepted"]
            result={"original_replay_id":original_id,"accepted":success,"source":source_provenance,
                "grid_plan":grid_summary,"execution":audit,
                "strict_admission":{k:v for k,v in admission.items() if k!="source_to_live_row"}}
            audit_path=args.out/"audit"/f"replay_{original_id:06d}.json"
            atomic_json(audit_path,{**result,"stats":stats,"all_source_events":events,
                "source_to_live_row":admission.get("source_to_live_row",[])})
            if success:
                arrays["flags"]=mask_omitted_commands(arrays["flags"],arrays["ticks"],arrays["seats"],
                    stats["omitted_command_ticks"],decision_ticks=args.decision_ticks)
                actions=arrays["actions"]
                illegal=~arrays["masks"][np.arange(len(actions)),actions].astype(bool)
                if np.any(illegal & (arrays["flags"]==0)):
                    raise ValueError("Fresh simulator produced unflagged illegal labels")
                if np.any(arrays["ticks"]%args.decision_ticks):
                    raise ValueError("Captured labels are not on the requested decision grid")
                arrays["replay_ids"]=np.zeros(len(actions),np.int32)
                compact_id=len(new_replays)
                path=args.out/"shards"/f"replay_{compact_id:06d}.npz"
                np.savez_compressed(path,**arrays)
                total_rows+=len(actions)
                shards.append({"path":str(path.relative_to(args.out)).replace("\\","/"),
                    "sha256":digest(path),"rows":len(actions),"replays":1,"replay_offset":compact_id})
                # Keep identity and split membership, but put the large repair ledger in audit/.
                compact={k:v for k,v in replay.items() if k not in {"repair_records","repair_summary"}}
                compact.update({k:v for k,v in stats.items() if k!="repair_records"})
                compact.update(original_replay_id=original_id,interval_audit=result,
                    repair_ledger_path=str(audit_path.relative_to(args.out)).replace("\\","/"),
                    source_backend_patch_sha256=replay.get("backend_patch_sha256",expected),
                    backend_patch_sha256=target_patch)
                new_replays.append(compact)
            results.append(result)
            del arrays
            print(json.dumps({"phase":"reconstructed","original_replay_id":original_id,
                "accepted":success,"accepted_games":len(new_replays),"rows":total_rows,
                "retention":audit["command_retention"]}),flush=True)
    summary={"selected_games":len(plans),"accepted_games":len(new_replays),"rows":total_rows,
        "decision_ticks":args.decision_ticks,"training_ready":False,
        "strict_subset_ready":bool(new_replays) and not args.allow_recorded_subset,
        "smoke_only":args.allow_recorded_subset,"results":results}
    source_count=sum(len(events) for _,events,_,_ in plans)
    retained_count=sum(result["strict_admission"]["source_commands"] for result in results if result["accepted"])
    summary.update(selected_source_commands=source_count,accepted_source_commands=retained_count,
        accepted_source_fraction=retained_count/max(1,source_count),
        all_selected_source_commands_preserved=retained_count==source_count)
    summary["training_ready"]=(bool(new_replays) and not args.allow_recorded_subset
        and summary["all_selected_source_commands_preserved"])
    atomic_json(args.out/"interval_summary.json",summary)
    if not new_replays: raise RuntimeError("No live replay passed source/terminal admission")
    metadata={k:v for k,v in original.items() if k not in {
        "replays","label_totals","rows","quality_tier_counts","supervised_replays","excluded_replay_ids"}}
    metadata.update(schema=vector_metadata["schema"],vocabulary=vector_metadata["vocabulary"],
        rows=total_rows,replays=new_replays,decision_ticks=args.decision_ticks,source_decision_ticks=original["decision_ticks"],
        interval_rebuild=protocol,training_ready=summary["training_ready"],smoke_only=args.allow_recorded_subset,
        strict_subset_ready=summary["strict_subset_ready"],
        all_selected_source_commands_preserved=summary["all_selected_source_commands_preserved"],
        label_source="live_pre_action_capture_interval_v1",backend_provenance=provenance,
        backend_patch_sha256=target_patch,source_backend_patch_sha256=expected,
        supervised_replays=len(new_replays),excluded_replay_ids=[])
    metadata["import_policy"]={**policy,"decision_ticks":args.decision_ticks,
        "time_tolerance_ticks":args.decision_ticks-1,"skip_unavailable_abilities":False,
        "serial_native_confirmation":False,"admission":"strict_all_source_commands_confirmed_supervised"}
    atomic_json(args.out/"dataset_metadata.json",metadata)
    atomic_json(args.out/"dataset.manifest.json",{"format":"crrl-bc-shards-v1","dataset_kind":"repaired",
        "rows":total_rows,"replays":len(new_replays),"decision_ticks":args.decision_ticks,
        "training_ready":summary["training_ready"],"smoke_only":args.allow_recorded_subset,
        "strict_subset_ready":summary["strict_subset_ready"],
        "all_selected_source_commands_preserved":summary["all_selected_source_commands_preserved"],
        "metadata":"dataset_metadata.json","metadata_sha256":digest(args.out/"dataset_metadata.json"),"shards":shards})


if __name__=="__main__":main()
