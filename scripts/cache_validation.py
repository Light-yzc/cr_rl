"""Publish compact raw-state validation caches and measure a verified SSD restore."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import zstandard

from cr_rl.data.persistent_cache import publish_validation_cache, restore_validation_cache
from cr_rl.training.persistence import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed signature directory on SSD")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--mode", choices=("publish", "benchmark"), default="benchmark")
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--low-priority", action="store_true")
    args = parser.parse_args()
    if args.low_priority and hasattr(os, "nice"):
        os.nice(10)
    record = json.loads((args.source / "complete.json").read_text())
    result = {
        "phase": "publishing",
        "observed_unix": time.time(),
        "signature": record["signature"],
        "dataset_sha256": record["dataset_sha256"],
        "context": record["context"],
        "zstandard_version": zstandard.__version__,
        "original_build_seconds": record.get("build_seconds"),
        "affects_training_data_or_model": False,
    }
    atomic_json(args.report, result)
    publication = publish_validation_cache(args.source, args.store, reserve_bytes=16 * 1024**3)
    metadata = json.loads((args.store / record["signature"] / "archive.json").read_text())
    result.update(publication=publication, archive=metadata, phase="published")
    atomic_json(args.report, result)
    print(json.dumps(result), flush=True)
    if args.mode == "benchmark":
        scratch = args.scratch or args.source.parent
        scratch.mkdir(parents=True, exist_ok=True)
        result["phase"] = "measuring_restore"
        atomic_json(args.report, result)
        with tempfile.TemporaryDirectory(prefix=".restore-probe-", dir=scratch) as temporary:
            restoration = restore_validation_cache(
                args.store,
                Path(temporary) / record["signature"],
                signature=record["signature"],
                dataset_sha256=record["dataset_sha256"],
                context=record["context"],
                maximum_bytes=metadata["uncompressed_bytes"],
                reserve_bytes=16 * 1024**3,
            )
        result.update(
            restoration=restoration,
            restore_conditions=(
                "Same VM immediately after verified upload; DriveFS/page caching may help. "
                "Not a fresh-VM network or whole-training restart benchmark."
            ),
            compression_fraction=metadata["archive_bytes"] / metadata["uncompressed_bytes"],
        )
    result.update(phase="completed", observed_unix=time.time())
    atomic_json(args.report, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
