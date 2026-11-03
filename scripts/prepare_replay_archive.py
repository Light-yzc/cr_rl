"""Wrap native replays for the upstream viewer's multi-match list.

Archive schema: crforge desktop/src/main/java/org/crforge/desktop/replay/ReplayArchive.java
@ 7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6 (https://github.com/voonhous/crforge).
Only the display copy's player names change; original replay files remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--policy-label")
    parser.add_argument("--opponent-label")
    args = parser.parse_args()
    files = sorted(
        p for p in args.directory.glob("match_*.json") if p.stem.removeprefix("match_").isdigit()
    )
    metadata = json.loads((args.directory / "evaluation.json").read_text(encoding="utf-8"))
    policy_label = args.policy_label or metadata.get("policy_label", "PPO")
    opponent_label = args.opponent_label or metadata.get("opponent_label", "BC")
    if len(files) != len(metadata["matchups"]):
        raise ValueError("Replay files and evaluation matchups do not agree")
    records = []
    single_views = args.directory / "viewer"
    single_views.mkdir(exist_ok=True)
    for index, path in enumerate(files):
        replay = json.loads(path.read_text(encoding="utf-8"))
        capture = replay.pop("capture")
        matchup = metadata["matchups"][index]
        for seat in (0, 1):
            role = policy_label if seat == matchup["learner_seat"] else opponent_label
            replay["battle"][f"avatar{seat}"]["name"] = role
        (single_views / path.name).write_text(
            json.dumps({**replay, "capture": capture}), encoding="utf-8"
        )
        record = {
            "format": 1,
            "session": {"client_version": capture.get("client_version")},
            "battle": {key: capture[key] for key in ("content_version", "content_sha")},
            "replay": json.dumps(replay, separators=(",", ":")),
            "cr_rl": {"match": index + 1, **matchup},
        }
        records.append(record)
    output = args.directory / "all_matches.jsonl"
    output.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records), encoding="utf-8"
    )
    print(f"Prepared {len(records)} viewer records: {output}")


if __name__ == "__main__":
    main()
