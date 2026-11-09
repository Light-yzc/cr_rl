"""Apply, publish or verify the registered crforge command-lookup patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cr_rl.config import ROOT, SimulatorPaths
from cr_rl.envs.backend_patch import apply_patches, installed_backend, write_build_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "publish", "verify"))
    parser.add_argument("--vendor", type=Path, default=ROOT / ".cache/vendor/crforge")
    parser.add_argument("--install", type=Path, default=SimulatorPaths.discover().install)
    args = parser.parse_args()
    if args.action == "apply":
        result = apply_patches(args.vendor)
    elif args.action == "publish":
        result = write_build_receipt(args.vendor, args.install)
    else:
        result = installed_backend(args.install)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
