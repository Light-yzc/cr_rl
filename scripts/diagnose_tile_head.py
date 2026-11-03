"""Bounded frozen-encoder placement-head ablation; does not modify training weights."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_bc import make_schema

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.bc_batch import build_bc_batch
from cr_rl.training.persistence import atomic_json


class PlacementProbe(nn.Module):
    def __init__(self, queries, attention, actor, residual=False):
        super().__init__()
        self.queries = nn.Parameter(queries.clone())
        self.attention = copy.deepcopy(attention)
        self.actor = copy.deepcopy(actor)
        self.residual = residual

    def forward(self, base, memory, mask, slots):
        query = self.queries[0, slots] + base[:, None]
        value, _ = self.attention(query, memory, memory, key_padding_mask=~mask, need_weights=False)
        if self.residual:
            value = value + query
        return self.actor(value).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--compare-precision", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    started = time.perf_counter()
    manifest = json.loads((args.dataset / "dataset.manifest.json").read_text())
    metadata = json.loads((args.dataset / manifest["metadata"]).read_text())
    with np.load(args.dataset / manifest["shards"][0]["path"]) as archive:
        obs, masks = archive["observations"], archive["masks"]
        actions, flags = archive["actions"], archive["flags"]
        replay_ids, seats, ticks = archive["replay_ids"], archive["seats"], archive["ticks"]
    valid = (
        (flags == 0)
        & (actions > 0)
        & (actions < 2305)
        & masks[np.arange(len(actions)), actions].astype(bool)
    )
    rows = np.random.default_rng(20261103).choice(
        np.flatnonzero(valid), args.samples, replace=False
    )
    groups = {}
    for i in range(len(actions)):
        groups.setdefault((int(replay_ids[i]), int(seats[i])), []).append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: int(ticks[i]))
    positions = {key: {row: i for i, row in enumerate(indices)} for key, indices in groups.items()}
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = ModelConfig(**state["config"])
    model = ActorCritic(cfg).cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    del state
    schema = make_schema(metadata)
    adapter = CrForgeObservationAdapter(
        schema, len(metadata["vocabulary"]["names"]), entity_bucket=16
    )
    bases, memories, entity_masks = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(rows), 2):
            ids = rows[offset : offset + 2]
            h, e, em, _, tm, hands, y, decks, seen = build_bc_batch(
                ids,
                obs,
                masks,
                actions,
                replay_ids,
                seats,
                groups,
                positions,
                adapter,
                cfg.context_len,
                "cuda",
            )
            context, memory, current_mask = model.encode_history(
                h,
                e,
                em,
                tm,
                deck_card_ids=decks,
                enemy_seen_card_ids=seen,
            )
            slot = ((y - 1) // 576).long()
            cards = model.card_embedding((hands.long() + 1).clamp(0, cfg.vocab_size))
            cards = cards + model.card_slot_embedding
            base = (
                model.tile_context(context)
                + cards[torch.arange(len(ids), device="cuda"), slot]
                + model.slot_embedding[0, slot, 0]
            )
            bases.append(base.float().cpu())
            memories.extend(x.float().cpu() for x in memory)
            entity_masks.extend(x.cpu() for x in current_mask)
    base = torch.cat(bases).cuda()
    memory = nn.utils.rnn.pad_sequence(memories, batch_first=True).cuda()
    entity_mask = nn.utils.rnn.pad_sequence(entity_masks, batch_first=True).cuda()
    target = torch.as_tensor((actions[rows] - 1) % 576, device="cuda", dtype=torch.long)
    slots = torch.as_tensor((actions[rows] - 1) // 576, device="cuda", dtype=torch.long)
    legal = torch.as_tensor(masks[rows, 1:2305].reshape(-1, 4, 576), device="cuda").bool()
    legal = legal[torch.arange(len(rows), device="cuda"), slots]
    prototype = PlacementProbe(
        model.tile_queries.detach(), model.tile_cross_attention, model.tile_actor
    )
    del model, obs, masks, groups, positions
    torch.cuda.empty_cache()

    def metrics(probe, bf16):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            raw = probe(base, memory, entity_mask, slots).float()
            logits = raw.masked_fill(~legal, -1e9)
            prediction = logits.argmax(-1)
            loss = F.cross_entropy(logits, target)
            spread = [float(raw[i, legal[i]].std().cpu()) for i in range(len(rows))]
            distance = (prediction % 18 - target % 18).abs() + (
                prediction // 18 - target // 18
            ).abs()
        return {
            "loss": float(loss.cpu()),
            "accuracy": float((prediction == target).float().mean().cpu()),
            "tile_logits_std": float(np.mean(spread)),
            "tile_l1": float(distance.float().mean().cpu()),
        }

    report = {
        "samples": len(rows),
        "encoder_parameters_frozen": True,
        "uniform_legal_tile_loss": float(legal.sum(-1).float().log().mean().cpu()),
        "entity_counts": entity_mask.sum(-1).cpu().tolist(),
        "cases": [],
    }
    cases = [
        (residual, lr, bf16)
        for residual in (False, True)
        for lr in (3e-5, 3e-4)
        for bf16 in ((False, True) if args.compare_precision else (True,))
    ]
    for residual, lr, bf16 in cases:
        probe = copy.deepcopy(prototype).cuda()
        probe.residual = residual
        case = {
            "bf16_head": bf16,
            "query_residual": residual,
            "learning_rate": lr,
            "trainable_parameters": sum(p.numel() for p in probe.parameters()),
            "initial_fp32": metrics(probe, False),
            "initial_bf16": metrics(probe, True),
            "history": [],
        }
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, fused=True)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = probe(base, memory, entity_mask, slots).float().masked_fill(~legal, -1e9)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            if step == 1:
                case["initial_query_grad_norm"] = float(probe.queries.grad.norm().cpu())
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in {1, 10, 50, 100, args.steps}:
                case["history"].append({"step": step, **metrics(probe, bf16)})
        report["cases"].append(case)
        atomic_json(args.out, report)
        print(json.dumps(case), flush=True)
        del optimizer, probe
        torch.cuda.empty_cache()
    report["seconds"] = time.perf_counter() - started
    report["limitation"] = (
        "Head-only fixed-batch overfit ablation, not a full-training or win-rate comparison."
    )
    atomic_json(args.out, report)


if __name__ == "__main__":
    main()
