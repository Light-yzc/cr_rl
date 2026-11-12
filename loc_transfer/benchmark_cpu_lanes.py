"""Compare simulator workers on identical seeded native games and legal requests."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from cr_rl.envs.crforge import CrForgeVector

results = []
for count in (2, 4, 8):
    expected = None
    for workers in (1, min(count, 4)):
        env = CrForgeVector(count, workers, independent=True)
        env.set_autoreset(False)
        for i in range(count):
            env.vector.reset_env(i, 4100 + i, env.deck, env.deck)
        rng = np.random.default_rng(42)
        digest = hashlib.sha256()
        elapsed = 0
        episodes = np.zeros(count, dtype=int)
        for step in range(1152):
            actions = np.array(
                [[rng.choice(np.flatnonzero(mask)) for mask in pair] for pair in env.masks],
                dtype=np.int32,
            )
            began = time.perf_counter()
            env.step(actions)
            duration = time.perf_counter() - began
            if step >= 128:
                elapsed += duration
            for key in ("observations", "masks", "rewards", "dones"):
                digest.update(getattr(env, key).tobytes())
            for i in np.flatnonzero(env.dones):
                episodes[i] += 1
                env.vector.reset_env(
                    int(i), int(4100 + i + episodes[i] * 10000), env.deck, env.deck
                )
        signature = digest.hexdigest()
        if expected is None:
            expected = signature
        assert signature == expected, "Concurrent native simulation changed state/mask/reward"
        env.close()
        row = {
            "envs": count,
            "workers": workers,
            "step_seconds": elapsed,
            "env_decisions_per_second": count * 1024 / elapsed,
            "exact_state_mask_reward_parity": True,
            "trajectory_sha256": signature,
        }
        print(json.dumps(row), flush=True)
        results.append(row)
Path("artifacts/local_dual_v100_checks/cpu_lane_benchmark.json").write_text(
    json.dumps(results, indent=2)
)
