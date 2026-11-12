"""Replay the two completed acceptance games from their recorded requests and seeds."""

import json
from pathlib import Path

import numpy as np
import torch

from cr_rl.envs.crforge import CrForgeVector

root = Path(__file__).resolve().parents[1]
out = root / "artifacts/local_dual_v100_checks"
paths = [out / "accepted_rollouts" / f"rollout_{u:06d}.npz" for u in (143, 144)]
archives = [np.load(p) for p in paths]
keys = (
    "obs",
    "masks",
    "actions",
    "next_obs",
    "next_masks",
    "rewards",
    "dones",
    "terminated",
    "fault_codes",
    "episode_ids",
    "seeds",
    "episode_start",
)
data = {key: np.concatenate([a[key] for a in archives]) for key in keys}
state = torch.load(
    root / "artifacts/local_dual_v100/ppo_latest.pt",
    map_location="cpu",
    mmap=True,
    weights_only=False,
)
league = state["league_config"]
env = CrForgeVector(1, 1, independent=True)
env.set_autoreset(False)
columns = list(env.vector.schema.header["deck"])
lookup = {
    tuple(env.vector.card_id(c) for c in cards): name for name, cards in league["decks"].items()
}
reports = []
try:
    for lane in range(data["obs"].shape[1]):
        episode = data["episode_ids"][0, lane]
        rows = np.flatnonzero(data["episode_ids"][:, lane] == episode)
        assert data["episode_start"][rows[0], lane]
        assert data["terminated"][rows[-1], lane]
        decks = [lookup[tuple(data["obs"][0, lane, seat, columns])] for seat in range(2)]
        seed = int(data["seeds"][0, lane])
        env.vector.reset_env(
            0,
            seed,
            league["decks"][decks[0]],
            league["decks"][decks[1]],
            league["slots"][decks[0]],
            league["slots"][decks[1]],
        )
        for row in rows:
            np.testing.assert_array_equal(env.observations[0], data["obs"][row, lane])
            np.testing.assert_array_equal(env.masks[0], data["masks"][row, lane])
            env.step(data["actions"][row, lane][None, :])
            np.testing.assert_array_equal(env.observations[0], data["next_obs"][row, lane])
            np.testing.assert_array_equal(env.masks[0], data["next_masks"][row, lane])
            np.testing.assert_array_equal(env.rewards[0], data["rewards"][row, lane])
            assert bool(env.dones[0]) == bool(data["dones"][row, lane])
            assert data["fault_codes"][row, lane] == 0
        reports.append(
            {
                "episode": int(episode),
                "seed": seed,
                "decks": decks,
                "decisions": len(rows),
                "exact_native_state_mask_reward_parity": True,
            }
        )
finally:
    env.close()
result = {
    "passed": True,
    "games": reports,
    "scope": "Replay of saved training requests; no new policy evaluation games",
}
(out / "native_rollout_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
