"""Verify immutable exported policies before seeding a resumed history league."""

import json
from pathlib import Path

import torch

from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.opponents.league import OpponentEntry
from cr_rl.opponents.role_sampling import historical_policy_id
from cr_rl.training.persistence import atomic_json, copy_verified, file_sha256


def seed_history(pool, manifest_path, output, config, provenance):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["policies"]
    if not rows or len({row["generation"] for row in rows}) != len(rows):
        raise ValueError("History seeds require unique generations")
    if any(e.role == "historical" for e in pool.entries):
        raise ValueError("Seed only a fresh history stage; resume its saved pool thereafter")
    if pool.max_entries and len(pool.entries) + len(rows) > pool.max_entries:
        raise ValueError("Seed set exceeds history pool capacity")
    checked = []
    for row in rows:
        path = Path(row["checkpoint"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if file_sha256(path) != row["sha256"]:
            raise ValueError("History seed checksum mismatch")
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        seed_config = ModelConfig(**state["config"])
        behavior_defaults = dict(
            frame_encoder_fp32=False, elixir_wait_bias=0.0, elixir_hold_min_cost=4.0
        )
        expected = dict(ModelConfig(**config).__dict__, **behavior_defaults)
        actual = dict(seed_config.__dict__, **behavior_defaults)
        if actual != expected or state.get("updates") != row["generation"]:
            raise ValueError("History seed architecture/generation mismatch")
        if any(state.get(k) != value for k, value in provenance.items()):
            raise ValueError("History seed simulator provenance mismatch")
        with torch.random.fork_rng(devices=[]):
            model = ActorPolicy(seed_config)
        model.load_state_dict(state["model"], strict=True)
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("Nonfinite history seed")
        checked.append((row, path))
        del state, model
    receipts = []
    for row, path in checked:
        policy_id = historical_policy_id(row["generation"])
        target = output / f"opponent_{policy_id:06d}.pt"
        receipt = copy_verified(path, target)
        if receipt["sha256"] != row["sha256"]:
            raise ValueError("History seed changed during verification")
        pool.add(
            OpponentEntry(
                policy_id,
                str(target.resolve()),
                row["generation"],
                deck_id="multi_deck",
                role="historical",
            )
        )
        receipts.append(dict(row, checkpoint=target.name, policy_id=policy_id))
    atomic_json(
        output / "history_bootstrap.json",
        {
            "manifest_sha256": file_sha256(manifest_path),
            "policies": receipts,
            "purpose": "historical opponents; no strongest-policy promotion",
        },
    )
