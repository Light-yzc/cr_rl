from __future__ import annotations

import argparse
import time

from cr_rl.envs.crforge import CrForgeVector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    for count in (1, 4, 8, 16, 32):
        with CrForgeVector(count=count, threads=min(count, 8)) as env:
            env.reset(list(range(20261102, 20261102 + count)))
            start = time.perf_counter()
            for _ in range(args.steps):
                env.actions[:] = 0
                env.step()
            elapsed = time.perf_counter() - start
            total = count * args.steps
            print(
                f"envs={count:2d} threads={min(count, 8):2d} "
                f"decisions={total:5d} elapsed={elapsed:.3f}s "
                f"throughput={total / elapsed:,.0f} env-decisions/s "
                f"faults={env.vector.fault_kind_counts()}"
            )


if __name__ == "__main__":
    main()
