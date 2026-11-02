from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict:
    with (ROOT / "configs" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@dataclass(frozen=True)
class SimulatorPaths:
    install: Path
    tables: Path

    @classmethod
    def discover(cls) -> SimulatorPaths:
        vendor = ROOT / ".cache/vendor/crforge"
        return cls(vendor / "env/build/install/env", Path.home() / ".crforge/tables/16.635.3")

    def validate(self) -> None:
        missing = [str(path) for path in (self.install, self.tables) if not path.exists()]
        if missing:
            raise FileNotFoundError("Simulator not bootstrapped; missing: " + ", ".join(missing))
