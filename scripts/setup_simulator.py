"""Build the simulator revision and game data pinned by configs/backends.yaml."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/backends.yaml").read_text())["primary"]
    vendor = ROOT / ".cache/vendor/crforge"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    if not (vendor / ".git").exists():
        run(["git", "clone", "--no-checkout", config["repository"], str(vendor)])
    run(["git", "fetch", "--depth", "1", "origin", config["commit"]], vendor)
    run(["git", "checkout", "--detach", config["commit"]], vendor)
    run(
        [
            "bash",
            "./gradlew",
            ":env:installDist",
            ":tables:dataVersionTables",
            f"-Pcrforge.assetSource={config['asset_source']}",
            f"-Pcrforge.dataVersion={config['data_version']}",
        ],
        vendor,
    )
    # uv-managed venvs intentionally do not include pip.
    import shutil

    if shutil.which("uv"):
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "-e",
                str(vendor / config["python_package"]),
            ]
        )
    else:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-e",
                str(vendor / config["python_package"]),
            ]
        )


if __name__ == "__main__":
    main()
