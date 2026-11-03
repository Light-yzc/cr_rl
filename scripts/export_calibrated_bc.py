"""Export weights with a single timing-bias correction for evaluation and PPO."""

import argparse
import json
from pathlib import Path

from cr_rl.training.calibrated_export import export_calibrated_bc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bias", type=float, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_calibrated_bc(
        args.checkpoint, args.bias, args.calibration_report, args.out
    )))


if __name__ == "__main__":
    main()
