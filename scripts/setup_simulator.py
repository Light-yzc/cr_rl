"""Build the simulator revision and game data pinned by configs/backends.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local", action="store_true", help="Reuse pinned checkout and cached Gradle"
    )
    parser.add_argument(
        "--skip-tables", action="store_true", help="Use already installed game tables"
    )
    args = parser.parse_args()
    from cr_rl.envs.backend_patch import apply_patches, write_build_receipt

    config = yaml.safe_load((ROOT / "configs/backends.yaml").read_text())["primary"]
    vendor = ROOT / ".cache/vendor/crforge"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    if args.local and not (vendor / ".git").exists():
        parser.error("--local requires the existing pinned simulator checkout")
    if not (vendor / ".git").exists():
        run(["git", "clone", "--no-checkout", config["repository"], str(vendor)])
    if not args.local:
        run(["git", "fetch", "--depth", "1", "origin", config["commit"]], vendor)
        run(["git", "checkout", "--detach", config["commit"]], vendor)
    apply_patches(vendor)
    command = ["bash", "./gradlew", ":env:installDist"]
    if not args.skip_tables:
        command.append(":tables:dataVersionTables")
    command += [
        f"-Pcrforge.assetSource={config['asset_source']}",
        f"-Pcrforge.dataVersion={config['data_version']}",
    ]
    if args.local:
        command.append("--offline")
    run(command, vendor)
    write_build_receipt(vendor, vendor / "env/build/install/env")
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
