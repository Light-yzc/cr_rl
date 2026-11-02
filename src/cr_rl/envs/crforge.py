from __future__ import annotations

import os
import subprocess
from pathlib import Path

from crforge_env import LadderConfig, VectorBattle

from cr_rl.config import SimulatorPaths, load_yaml


class CrForgeVector:
    """Thin, version-pinned adapter. Raw simulator schema stays intact at this boundary."""

    def __init__(self, count: int = 1, threads: int = 1, paths: SimulatorPaths | None = None):
        # macOS may still register an older JDK with java_home while Homebrew's JDK 21
        # is installed keg-only. JPype must see the intended libjvm before it starts.
        brew_java = Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
        if brew_java.joinpath("lib/server/libjvm.dylib").exists():
            os.environ["JAVA_HOME"] = str(brew_java)
        elif not os.environ.get("JAVA_HOME"):
            os.environ["JAVA_HOME"] = subprocess.check_output(
                ["/usr/libexec/java_home", "-v", "21"], text=True
            ).strip()
        self.paths = paths or SimulatorPaths.discover()
        self.paths.validate()
        deck_cfg = load_yaml("deck.yaml")
        self.deck = deck_cfg["cards"]
        self.level = int(deck_cfg["level"])
        self.vector = VectorBattle(
            count,
            threads,
            install=str(self.paths.install),
            tables=str(self.paths.tables),
            config=LadderConfig(
                decision_ticks=5,
                level=self.level,
                tower_level=int(deck_cfg["tower_level"]),
                fault_bundles=str(Path("artifacts/fault-bundles").resolve()),
            ),
        )
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
    ):
        self.vector.reset(
            seeds or list(range(self.count)),
            deck0 or self.deck,
            deck1 or self.deck,
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
        return dict(self.vector.metadata)

    def close(self) -> None:
        self.vector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
