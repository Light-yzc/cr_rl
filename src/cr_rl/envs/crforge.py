from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    ):
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
            if self.backend_commit != expected_commit:
                raise ValueError(f"Simulator revision differs from pinned {expected_commit}")
        deck_cfg = load_yaml("deck.yaml")
        self.deck = deck_cfg["cards"]
        self.level = int(deck_cfg["level"])

        def factory(n, t):
            return VectorBattle(
                n,
                t,
                install=str(self.paths.install),
                tables=str(self.paths.tables),
                config=LadderConfig(
                    decision_ticks=5,
                    level=self.level,
                    tower_level=int(deck_cfg["tower_level"]),
                    fault_bundles=str(Path("artifacts/fault-bundles").resolve()),
                    arena=arena,
                ),
            )

        self.vector = IndependentBattles(count, factory) if independent else factory(count, threads)
        from cr_rl.envs.backend_patch import loaded_backend

        try:
            self.backend_provenance = loaded_backend(self.paths.install)
        except Exception:
            self.vector.close()
            raise
        self.count = count

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
            "backend_provenance": self.backend_provenance,
            "backend_patch_sha256": self.backend_provenance["patch_manifest_sha256"],
        }

    def close(self) -> None:
        self.vector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class IndependentBattles:
    """Public-API-only lanes: change one match's deck without resetting other matches.

    The pinned VectorBattle.reset_env keeps decks. Separate one-match vectors let us
    use its supported reset(decks, slots) API for per-episode curriculum sampling.
    """

    def __init__(self, count, factory):
        self.lanes = [factory(1, 1) for _ in range(count)]
        for name in ("observations", "masks", "actions", "rewards", "dones", "infos"):
            example = getattr(self.lanes[0], name)
            setattr(self, name, np.concatenate([example.copy() for _ in range(count)], axis=0))

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

    def reset_env(self, index, seed, deck0=None, deck1=None, slots0=None, slots1=None):
        if deck0 is None:
            self.lanes[index].reset_env(0, seed)
        else:
            self.lanes[index].reset([seed], deck0, deck1, slots0, slots1)
        self._sync(index)

    def step(self):
        for index, lane in enumerate(self.lanes):
            lane.actions[:] = self.actions[index : index + 1]
            lane.step()
            self._sync(index)

    def info(self, index):
        return self.lanes[index].info(0)

    def episode_seed(self, index):
        return self.lanes[index].episode_seed(0)

    def set_autoreset(self, enabled):
        for lane in self.lanes:
            lane.set_autoreset(enabled)

    def close(self):
        for lane in self.lanes:
            lane.close()
