"""Numerical smoke for the v3 recurrent PPO update on an existing checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.state_ppo import StatePPOConfig, state_ppo_update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v3 PPO smoke")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("model_architecture_version") != 3 or checkpoint.get("decision_ticks") != 8:
        raise ValueError("Smoke requires a v3 checkpoint trained at 8 ticks")
    device = torch.device(args.device)
    model = state_model_from_checkpoint(checkpoint, device=device)
    model.frame_microbatch = 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=0.0, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    cfg = model.cfg
    batch, steps, entities = 1, 4, min(32, cfg.max_entities)
    generator = torch.Generator(device=device).manual_seed(20261113)
    sequences = []
    for _ in range(2):
        header = torch.randn(steps, cfg.header_dim, device=device, generator=generator)
        entity = torch.randn(steps, entities, cfg.entity_feature_dim, device=device, generator=generator)
        entity[..., 4:6] = entity[..., 4:6].sigmoid()
        mask = torch.ones(steps, entities, dtype=torch.bool, device=device)
        legal = torch.ones(steps, cfg.action_count, dtype=torch.bool, device=device)
        hand = torch.randint(0, cfg.vocab_size - 1, (steps, 4), device=device, generator=generator)
        with torch.no_grad():
            state = model.initial_state(1, device=device)
            old, _, _ = model.forward_state(header, entity, mask, legal_action_mask=legal,
                                            state=state, hand_card_ids=hand, compute_value=True)
            actions = torch.multinomial(old.float().exp(), 1, generator=generator).squeeze(-1)
            old_logp = old.gather(-1, actions[:, None]).squeeze(-1).float()
        sequences.append({"header": header, "entities": entity, "entity_mask": mask,
                          "legal_action_mask": legal, "hand_card_ids": hand,
                          "actions": actions, "old_logp": old_logp,
                          "advantages": torch.randn(steps, device=device, generator=generator),
                          "returns": torch.randn(steps, device=device, generator=generator)})
    results = []
    for step in range(args.steps):
        result = state_ppo_update(model, optimizer, sequences,
                                  config=StatePPOConfig(clip_coef=0.2, value_coef=0.5,
                                                        entropy_coef=0.0001, precision=args.precision),
                                  scaler=scaler if args.precision == "fp16" else None)
        results.append(result)
        print(json.dumps({"step": step + 1, **result}), flush=True)
    print(json.dumps({"status": "pass", "architecture": 3, "decision_ticks": 8,
                      "precision": args.precision, "steps": len(results)}), flush=True)


if __name__ == "__main__":
    main()
