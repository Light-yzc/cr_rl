from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from crforge_env import LadderConfig, VectorBattle

from cr_rl.config import SimulatorPaths, load_yaml


class CrForgeVector:
    """Thin, version-pinned adapter. Raw simulator schema stays intact at this boundary."""

    def __init__(
        self,
        count: int = 1,
        threads: int = 1,
        paths: SimulatorPaths | None = None,
        independent: bool = False,
        arena: str | None = None,
        level_mixture: list[int] | tuple[int, ...] | None = None,
        tower_level_mixture: list[int] | tuple[int, ...] | None = None,
        level_jitter_prob: float | None = None,
        level_jitter: int = 2,
        level_jitter_seed: int = 0,
        decision_ticks: int = 5,
    ):
        if decision_ticks < 1:
            raise ValueError("Decision interval must contain positive simulator ticks")
        # macOS may still register an older JDK with java_home while Homebrew's JDK 21
        # is installed keg-only. JPype must see the intended libjvm before it starts.
        brew_java = Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
        if brew_java.joinpath("lib/server/libjvm.dylib").exists():
            os.environ["JAVA_HOME"] = str(brew_java)
        elif not os.environ.get("JAVA_HOME") and sys.platform == "darwin":
            os.environ["JAVA_HOME"] = subprocess.check_output(
                ["/usr/libexec/java_home", "-v", "21"], text=True
            ).strip()
        elif not os.environ.get("JAVA_HOME") and sys.platform.startswith("linux"):
            java = shutil.which("java")
            if java:
                os.environ["JAVA_HOME"] = str(Path(java).resolve().parents[1])
        self.paths = paths or SimulatorPaths.discover()
        self.paths.validate()
        vendor = self.paths.install.resolve().parents[3]
        self.backend_commit = None
        expected_commit = load_yaml("backends.yaml")["primary"]["commit"]
        if (vendor / ".git").exists():
            self.backend_commit = subprocess.check_output(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True
            ).strip()
        elif (vendor / 'PACKAGED_REVISION').is_file():
            self.backend_commit = (vendor / 'PACKAGED_REVISION').read_text().strip()
        if self.backend_commit is not None and self.backend_commit != expected_commit:
            raise ValueError(f"Simulator revision differs from pinned {expected_commit}")
        deck_cfg = load_yaml("deck.yaml")
        self.deck = deck_cfg["cards"]
        self.level = int(deck_cfg["level"])
        self.tower_level = int(deck_cfg["tower_level"])
        base_levels = _normalize_level_mixture(level_mixture, self.level, "card")
        if level_jitter_prob is not None:
            expanded_base = [base_levels[index % len(base_levels)] for index in range(count)]
            self.level_mixture = _apply_level_jitter(
                expanded_base,
                level_jitter_prob,
                level_jitter,
                level_jitter_seed,
            )
        else:
            self.level_mixture = base_levels
        if tower_level_mixture is None and len(self.level_mixture) > 1:
            # A level curriculum normally varies cards and towers together.  An explicit
            # tower schedule can still hold towers fixed or use a different pairing.
            tower_level_mixture = self.level_mixture
        self.tower_level_mixture = _normalize_level_mixture(
            tower_level_mixture,
            self.tower_level,
            "tower",
            expected_len=len(self.level_mixture),
        )
        if self.level_mixture != [self.level] and not independent:
            raise ValueError(
                "A level mixture requires independent=True so each lane can use its own global level"
            )
        if self.tower_level_mixture != [self.tower_level] and not independent:
            raise ValueError(
                "A tower level mixture requires independent=True so each lane can use its own global tower level"
            )

        def factory(n, t, lane_index=0):
            card_level = self.level_mixture[lane_index % len(self.level_mixture)]
            tower_level = self.tower_level_mixture[lane_index % len(self.tower_level_mixture)]
            return VectorBattle(
                n,
                t,
                install=str(self.paths.install),
                tables=str(self.paths.tables),
                config=LadderConfig(
                    decision_ticks=decision_ticks,
                    level=card_level,
                    tower_level=tower_level,
                    fault_bundles=str(Path("artifacts/fault-bundles").resolve()),
                    arena=arena,
                ),
            )

        self.vector = (
            IndependentBattles(
                count,
                factory,
                threads,
                level_mixture=(self.level_mixture if level_mixture is not None else None),
                tower_level_mixture=(
                    self.tower_level_mixture if tower_level_mixture is not None else None
                ),
            )
            if independent
            else factory(count, threads)
        )
        self.count = count
        self.levels = tuple(
            self.level_mixture[index % len(self.level_mixture)] for index in range(count)
        )
        self.tower_levels = tuple(
            self.tower_level_mixture[index % len(self.tower_level_mixture)]
            for index in range(count)
        )
        self.level_profile = {
            "mode": "per_environment_global",
            "card_levels": list(self.levels),
            "tower_levels": list(self.tower_levels),
            "backend_semantics": "one global card/tower level per environment lane; both seats share it",
            "jitter": (
                {
                    "probability": float(level_jitter_prob),
                    "amount": int(level_jitter),
                    "seed": int(level_jitter_seed),
                }
                if level_jitter_prob is not None
                else None
            ),
        }

    @property
    def observations(self):
        return self.vector.observations

    @property
    def masks(self):
        return self.vector.masks

    @property
    def actions(self):
        return self.vector.actions

    @property
    def rewards(self):
        return self.vector.rewards

    @property
    def dones(self):
        return self.vector.dones

    @property
    def infos(self):
        return self.vector.infos

    def reset(
        self,
        seeds: list[int] | None = None,
        deck0=None,
        deck1=None,
        slots0=None,
        slots1=None,
    ):
        self.vector.reset(
            seeds or list(range(self.count)),
            deck0 or self.deck,
            deck1 or self.deck,
            slots0,
            slots1,
        )
        return self.observations, self.masks

    def reset_env(self, index: int, seed: int) -> None:
        self.vector.reset_env(index, seed)

    def set_autoreset(self, enabled: bool) -> None:
        self.vector.set_autoreset(enabled)

    def step(self, actions=None) -> None:
        if actions is not None:
            self.actions[:] = actions
        self.vector.step()

    def metadata(self) -> dict:
        return {
            **dict(self.vector.metadata),
            "commit": self.backend_commit,
            "level_profile": self.level_profile,
        }

    def close(self) -> None:
        self.vector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _normalize_level_mixture(values, default: int, name: str, expected_len: int | None = None):
    """Validate a lane-level schedule without pretending the backend supports per-card levels."""
    if values is None:
        result = [int(default)]
    else:
        result = [int(value) for value in values]
        if not result:
            raise ValueError(f"{name} level mixture cannot be empty")
    if any(value < 1 or value > 15 for value in result):
        raise ValueError(f"{name} levels must be in [1, 15], got {result}")
    if expected_len is not None and len(result) not in (1, expected_len):
        raise ValueError(
            f"tower level mixture must contain one value or exactly {expected_len} values, got {len(result)}"
        )
    return result


def _apply_level_jitter(
    base_levels: list[int],
    probability: float,
    amount: int,
    seed: int,
) -> list[int]:
    """Perturb a reproducible minority of lane base levels by exactly ±amount."""
    if not base_levels:
        raise ValueError("At least one environment lane is required")
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"level jitter probability must be in [0, 1], got {probability}")
    if amount < 1:
        raise ValueError(f"level jitter amount must be positive, got {amount}")
    if any(not 1 <= base <= 15 for base in base_levels):
        raise ValueError(f"base levels must be in [1, 15], got {base_levels}")
    rng = np.random.default_rng(seed)
    result = [
        int(np.clip(base + (amount if rng.integers(2) else -amount), 1, 15))
        if rng.random() < probability
        else int(base)
        for base in base_levels
    ]
    # A small probability with only a few lanes can otherwise produce no variant at all.
    if probability > 0 and len(base_levels) > 1 and result == base_levels:
        index = int(rng.integers(len(base_levels)))
        base = base_levels[index]
        result[index] = int(np.clip(base + (amount if rng.integers(2) else -amount), 1, 15))
    return result


class IndependentBattles:
    """Public-API-only lanes: change one match's deck without resetting other matches.

    The pinned VectorBattle.reset_env keeps decks. Separate one-match vectors let us
    use its supported reset(decks, slots) API for per-episode curriculum sampling.
    A persistent outer pool runs native steps in parallel; each task owns one lane.
    """

    def __init__(self, count, factory, threads=1, level_mixture=None, tower_level_mixture=None):
        if count < 1 or threads < 1:
            raise ValueError("Positive environment and thread counts are required")
        self.threads = min(count, threads)
        diagnostic = os.getenv("CRRL_NATIVE_STEP_DIAGNOSTIC_SECONDS") if self.threads > 1 else None
        self._step_diagnostic_seconds = float(diagnostic) if diagnostic else None
        if self._step_diagnostic_seconds is not None and not (
            np.isfinite(self._step_diagnostic_seconds) and self._step_diagnostic_seconds > 0
        ):
            raise ValueError("CRRL_NATIVE_STEP_DIAGNOSTIC_SECONDS must be finite and positive")
        self._diagnostic_reset_seeds = [None] * count if diagnostic else None
        self._executor = (
            ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="battle-lane")
            if self.threads > 1 else None
        )
        if level_mixture is None and tower_level_mixture is None:
            self.lanes = [factory(1, 1) for _ in range(count)]
        else:
            self.lanes = [factory(1, 1, index) for index in range(count)]
        for name in ("observations", "masks", "actions", "rewards", "dones", "infos"):
            example = getattr(self.lanes[0], name)
            setattr(self, name, np.concatenate([example.copy() for _ in range(count)], axis=0))
        if diagnostic:
            schema = getattr(self.lanes[0], "schema", None)
            self._diagnostic_tick_column = getattr(schema, "info", {}).get("tick")

    def __getattr__(self, name):
        return getattr(self.lanes[0], name)

    def _sync(self, index):
        for name in ("observations", "masks", "rewards", "dones", "infos"):
            getattr(self, name)[index] = getattr(self.lanes[index], name)[0]

    def reset(self, seeds, deck0, deck1, slots0=None, slots1=None):
        for index, seed in enumerate(seeds):
            self.reset_env(
                index,
                seed,
                deck0[index],
                deck1[index],
                None if slots0 is None else slots0[index],
                None if slots1 is None else slots1[index],
            )

    def reset_env(self, index, seed, deck0=None, deck1=None, slots0=None, slots1=None, towers=None):
        if towers is not None and hasattr(self.lanes[index], "set_towers"):
            self.lanes[index].set_towers(0, towers[0], towers[1])
        if deck0 is None:
            self.lanes[index].reset_env(0, seed)
        else:
            self.lanes[index].reset([seed], deck0, deck1, slots0, slots1)
        self._sync(index)
        if self._diagnostic_reset_seeds is not None:
            self._diagnostic_reset_seeds[index] = int(seed)

    def fork_from_lane(self, source_vector, source_index: int, target_indices) -> None:
        targets = [int(i) for i in target_indices]
        if not targets:
            return
        src_lane = source_vector.lanes[int(source_index)]
        for dst_idx in targets:
            self.lanes[dst_idx].fork_from(src_lane, 0, -1, [0])
            self._sync(dst_idx)

    def fork_tasks_parallel(self, fork_groups) -> None:
        if not fork_groups:
            return
        grouped: dict[tuple[int, int], tuple[object, int, list[int]]] = {}
        for src_vec, src_idx, targets in fork_groups:
            key = (id(src_vec), int(src_idx))
            if key not in grouped:
                grouped[key] = (src_vec, int(src_idx), [])
            grouped[key][2].extend(int(i) for i in targets)
        items = list(grouped.values())
        if self._executor is None or len(items) <= 1:
            for src_vec, src_idx, targets in items:
                self.fork_from_lane(src_vec, src_idx, targets)
            return
        futures = [
            self._executor.submit(self.fork_from_lane, src_vec, src_idx, targets)
            for src_vec, src_idx, targets in items
        ]
        for fut in futures:
            fut.result()

    def _step_lane(self, index):
        lane = self.lanes[index]
        lane.actions[:] = self.actions[index : index + 1]
        lane.step()
        self._sync(index)

    def _step_batch(self, batch):
        for index in batch:
            lane = self.lanes[index]
            lane.actions[:] = self.actions[index : index + 1]
            lane.step()
            self._sync(index)

    def _report_slow_step(self, pending_lanes):
        frames = sys._current_frames()
        stacks = []
        caller = threading.current_thread()
        pending = set(pending_lanes)
        for thread in threading.enumerate():
            frame = frames.get(thread.ident)
            worker_frame = frame
            lane = None
            while worker_frame is not None:
                if (worker_frame.f_code is IndependentBattles._step_lane.__code__
                        and worker_frame.f_locals.get("self") is self):
                    lane = worker_frame.f_locals.get("index")
                    break
                worker_frame = worker_frame.f_back
            if frame is not None and (thread is caller or lane in pending):
                stacks.append(dict(thread=thread.name, native_id=thread.native_id, lane=lane,
                                   stack=traceback.format_stack(frame, limit=12)))
                if len(stacks) == 16:
                    break
        tick = self._diagnostic_tick_column
        print(json.dumps(dict(event="native_step_slow", timeout_seconds=self._step_diagnostic_seconds,
            pending_lanes=[dict(lane=i, reset_seed=self._diagnostic_reset_seeds[i],
                last_synced_tick=int(self.infos[i, tick]) if tick is not None else None,
                actions=self.actions[i].tolist()) for i in pending_lanes],
            python_stacks=stacks)), file=sys.stderr, flush=True)

    def step_lanes(self, indices):
        targets = [int(index) for index in indices]
        if not targets:
            return
        if self._executor is None or len(targets) == 1:
            self._step_batch(targets)
            return
        n_chunks = min(len(targets), self.threads * 2)
        chunks = [targets[i::n_chunks] for i in range(n_chunks)]
        futures = [self._executor.submit(self._step_batch, ch) for ch in chunks]
        # Join every lane on failure: reset/close must not race an unfinished step.
        _, pending = wait(futures, timeout=self._step_diagnostic_seconds)
        if pending:
            try:
                self._report_slow_step([idx for ch, future in zip(chunks, futures) if future in pending for idx in ch])
            except Exception:
                # A failed diagnostic must not bypass the join or mask a lane's exception.
                pass
            wait(pending)
        for future in futures:
            future.result()

    def step(self):
        self.step_lanes(range(len(self.lanes)))

    def info(self, index):
        return self.lanes[index].info(0)

    def episode_seed(self, index):
        return self.lanes[index].episode_seed(0)

    def set_autoreset(self, enabled):
        for lane in self.lanes:
            lane.set_autoreset(enabled)

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        for lane in self.lanes:
            lane.close()
