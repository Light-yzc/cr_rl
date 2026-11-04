"""Complete seeded legal-random games and replay saved requests against state digests.

This checks the pinned simulator interface, not fidelity to an official client or
policy strength. Compact traces store every request, both seat headers and exact
pre/post state+mask digests; they are not human demonstrations or PPO rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from cr_rl.envs.crforge import CrForgeVector
from cr_rl.training.persistence import atomic_json, file_sha256


def fingerprint(observation, mask):
    digest = hashlib.sha256(np.asarray(observation, dtype="<i4").tobytes())
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
    return np.frombuffer(digest.digest(), np.uint8).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--replay-checks", type=int, default=3)
    parser.add_argument("--decks", type=Path, default=Path("configs/decks_initial3.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.games, args.envs, args.max_decisions, args.replay_checks) < 1:
        parser.error("Game, lane, horizon and replay-check counts must be positive")
    if args.out.exists():
        parser.error("Use a fresh output directory to preserve acceptance evidence")
    args.out.mkdir(parents=True)
    config = yaml.safe_load(args.decks.read_text())
    names = list(config["decks"])
    games = [
        dict(
            game=index,
            seed=args.seed + index,
            deck0=names[index % len(names)],
            deck1=names[(index // len(names)) % len(names)],
        )
        for index in range(args.games)
    ]
    count = min(args.envs, args.games)
    rngs = [np.random.default_rng(game["seed"] + 100000) for game in games]
    pending = iter(games)
    active = [next(pending, None) for _ in range(count)]
    traces = [None] * count
    finished = []
    decisions = 0
    began = time.perf_counter()

    def decks(game):
        return (
            config["decks"][game["deck0"]],
            config["decks"][game["deck1"]],
            config.get("slots", {}).get(game["deck0"], [0] * 8),
            config.get("slots", {}).get(game["deck1"], [0] * 8),
        )

    def empty_trace():
        return {
            key: []
            for key in (
                "actions",
                "headers",
                "next_headers",
                "pre_digest",
                "post_digest",
                "rewards",
            )
        }

    try:
        with CrForgeVector(count, count, independent=True, arena="Arena_L") as env:
            metadata = env.metadata()
            atomic_json(
                args.out / "protocol.json",
                {
                    "command": vars(args) | {"decks": str(args.decks), "out": str(args.out)},
                    "decks_sha256": file_sha256(args.decks),
                    "metadata": metadata,
                    "policy": "independent seeded uniform over legal joint actions per seat",
                    "purpose": "interface acceptance only; not official-client fidelity",
                },
            )
            env.set_autoreset(False)
            for lane, game in enumerate(active):
                env.vector.reset_env(lane, game["seed"], *decks(game))
                traces[lane] = empty_trace()
            while any(active):
                actions = np.zeros((count, 2), np.int32)
                for lane, game in enumerate(active):
                    if game is None:
                        continue
                    trace = traces[lane]
                    if len(trace["actions"]) >= args.max_decisions:
                        raise RuntimeError(f"Game {game['game']} exceeded decision limit")
                    for seat in range(2):
                        legal = np.flatnonzero(env.masks[lane, seat])
                        if not len(legal):
                            raise RuntimeError("No legal action")
                        actions[lane, seat] = rngs[game["game"]].choice(legal)
                    trace["actions"].append(actions[lane].copy())
                    trace["headers"].append(
                        env.observations[lane, :, : env.vector.schema.header_ints].copy()
                    )
                    trace["pre_digest"].append(fingerprint(env.observations[lane], env.masks[lane]))
                    decisions += 1
                if not np.take_along_axis(env.masks, actions[..., None], -1)[..., 0].all():
                    raise RuntimeError("Sampled an illegal action")
                env.step(actions)
                for lane, game in enumerate(active):
                    if game is None:
                        continue
                    trace = traces[lane]
                    trace["post_digest"].append(
                        fingerprint(env.observations[lane], env.masks[lane])
                    )
                    trace["next_headers"].append(
                        env.observations[lane, :, : env.vector.schema.header_ints].copy()
                    )
                    trace["rewards"].append(env.rewards[lane].copy())
                    if not np.isfinite(env.rewards[lane]).all() or not np.isclose(
                        env.rewards[lane].sum(), 0
                    ):
                        raise RuntimeError("Rewards are nonfinite or not zero sum")
                    info = env.vector.info(lane)
                    if info["fault"] or info["truncated"]:
                        raise RuntimeError(f"Invalid game {game['game']}: {info}")
                    if not env.dones[lane]:
                        continue
                    if info["winner"] not in (0, 1, 2):
                        raise RuntimeError("A terminal game has no valid outcome")
                    prefix = args.out / f"game_{game['game']:04d}"
                    np.savez_compressed(
                        prefix.with_suffix(".npz"),
                        **{key: np.stack(value) for key, value in trace.items()},
                    )
                    prefix.with_suffix(".replay.json").write_text(
                        env.vector.lanes[lane].replay_json(0)
                    )
                    result = dict(game, info=info, decisions=len(trace["actions"]))
                    result["trace_sha256"] = file_sha256(prefix.with_suffix(".npz"))
                    atomic_json(prefix.with_suffix(".json"), result)
                    finished.append(result)
                    active[lane] = next(pending, None)
                    if active[lane] is not None:
                        env.vector.reset_env(lane, active[lane]["seed"], *decks(active[lane]))
                        traces[lane] = empty_trace()
                    else:
                        env.reset_env(lane, args.seed + 1000000 + lane)
                    if len(finished) % 10 == 0 or len(finished) == args.games:
                        print(
                            json.dumps({"completed_games": len(finished), "decisions": decisions}),
                            flush=True,
                        )
        replayed = []
        with CrForgeVector(1, 1, independent=True, arena="Arena_L") as env:
            env.set_autoreset(False)
            for game in sorted(finished, key=lambda row: row["game"])[: args.replay_checks]:
                env.vector.reset_env(0, game["seed"], *decks(game))
                path = args.out / f"game_{game['game']:04d}.npz"
                with np.load(path) as trace:
                    for index, actions in enumerate(trace["actions"]):
                        np.testing.assert_array_equal(
                            fingerprint(env.observations[0], env.masks[0]),
                            trace["pre_digest"][index],
                        )
                        env.step(actions[None])
                        np.testing.assert_array_equal(
                            fingerprint(env.observations[0], env.masks[0]),
                            trace["post_digest"][index],
                        )
                        np.testing.assert_array_equal(env.rewards[0], trace["rewards"][index])
                    if not env.dones[0] or env.vector.info(0) != game["info"]:
                        raise RuntimeError("Replayed terminal result differs")
                replayed.append(game["game"])
        elapsed = time.perf_counter() - began
        report = {
            "phase": "completed",
            "requested_games": args.games,
            "terminal_games": len(finished),
            "faults": 0,
            "truncations": 0,
            "illegal_actions": 0,
            "decisions": decisions,
            "seconds_including_save_and_replay": elapsed,
            "decisions_per_second_including_save_and_replay": decisions / elapsed,
            "bitwise_replayed_games": replayed,
            "games": finished,
            "limitation": "Simulator self-consistency; no official-client comparison",
        }
        atomic_json(args.out / "acceptance.json", report)
        print(
            json.dumps({key: value for key, value in report.items() if key != "games"}), flush=True
        )
    except Exception as error:
        atomic_json(
            args.out / "acceptance.json",
            {
                "phase": "failed",
                "completed_games": len(finished),
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
