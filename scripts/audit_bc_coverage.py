"""Audit retained BC trajectories without reading observations or touching array caches.

Durations describe saved observation intervals, not verified complete matches.
Source ``used`` counts are not substituted for legal, unmasked training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def replay_identity(replay):
    return replay.get("canonical_game_signature") or json.dumps(
        [replay.get("source"), replay.get("source_revision"), replay["replay_tag"]]
    )


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def quality_tier(
    record,
    *,
    a_horizon_seconds=60.0,
    a_min_labels=8,
    b_horizon_seconds=30.0,
    b_min_labels=4,
):
    """Classify one retained replay for BC/PPO use.

    This is deliberately based on native labelled rows and the retained prefix,
    rather than source ``used`` counts.  The returned tier is a data quality
    admission signal, not a claim that the reconstructed state is identical to
    the original client state.
    """
    if record["excluded"]:
        return "excluded"
    horizon = record["retained_horizon_seconds"]
    labels = record["valid_card_plays"] + record["valid_abilities"]
    paired = record["complete_paired_history_grid"] and record["observed_seats"] == 2
    # Unknown provenance must fail closed.  A missing field is common in old
    # fixtures, but it is not evidence that suffix rows were removed.
    bounded = record.get("reconstruction_policy") in {
        "first_omission_prefix",
        "bounded_omission_prefix",
    }
    collisions = record.get("collisions", 0)
    if (
        paired
        and bounded
        and collisions == 0
        and horizon is not None
        and horizon >= a_horizon_seconds
        and labels >= a_min_labels
    ):
        return "A"
    if (
        paired
        and bounded
        and horizon is not None
        and horizon >= b_horizon_seconds
        and labels >= b_min_labels
    ):
        return "B"
    return "C"


def _summary(
    records,
    late_seconds,
    *,
    a_horizon_seconds=60.0,
    a_min_labels=8,
    b_horizon_seconds=30.0,
    b_min_labels=4,
):
    active = [row for row in records if not row["excluded"]]
    late = sum(
        row["retained_horizon_seconds"] is not None
        and row["retained_horizon_seconds"] >= late_seconds
        for row in active
    )
    observed_late = sum(
        row["last_observation_seconds"] is not None
        and row["last_observation_seconds"] >= late_seconds
        for row in active
    )
    tiers = {
        tier: sum(
            quality_tier(
                row,
                a_horizon_seconds=a_horizon_seconds,
                a_min_labels=a_min_labels,
                b_horizon_seconds=b_horizon_seconds,
                b_min_labels=b_min_labels,
            )
            == tier
            for row in active
        )
        for tier in ("A", "B", "C")
    }
    return {
        "raw_metadata_games": len(records),
        "excluded_metadata_games": len(records) - len(active),
        "active_metadata_games": len(active),
        "active_games_with_observations": sum(row["observation_rows"] > 0 for row in active),
        "active_games_with_valid_supervision": sum(row["valid_rows"] > 0 for row in active),
        "active_games_with_valid_card_plays": sum(row["valid_card_plays"] > 0 for row in active),
        "active_games_missing_one_or_both_seat_histories": sum(
            row["observed_seats"] < 2 for row in active
        ),
        "active_games_with_complete_paired_history_grid": sum(
            row["complete_paired_history_grid"] for row in active
        ),
        "retained_history_rows": sum(row["observation_rows"] for row in active),
        "valid_supervision_rows": sum(row["valid_rows"] for row in active),
        "valid_card_play_labels": sum(row["valid_card_plays"] for row in active),
        "valid_ability_labels": sum(row["valid_abilities"] for row in active),
        "valid_wait_labels": sum(row["valid_waits"] for row in active),
        "flagged_rows": sum(row["flagged_rows"] for row in active),
        "unflagged_illegal_target_rows": sum(row["unflagged_illegal_rows"] for row in active),
        "quality_tiers": tiers,
        "ppo_bootstrap_eligible_games": tiers["A"],
        "ppo_bootstrap_eligible_fraction": tiers["A"] / len(active) if active else 0.0,
        "quality_thresholds": {
            "A_horizon_seconds": a_horizon_seconds,
            "A_min_labels": a_min_labels,
            "B_horizon_seconds": b_horizon_seconds,
            "B_min_labels": b_min_labels,
        },
        "retained_horizon_seconds": _distribution(
            [row["retained_horizon_seconds"] for row in active]
        ),
        "retained_observed_span_seconds": _distribution(
            [row["retained_span_seconds"] for row in active]
        ),
        "observed_interval_seconds": _distribution(
            [row["observed_interval_seconds"] for row in active]
        ),
        "observed_interval_duration_at_least_threshold_games": sum(
            row["observed_interval_seconds"] >= late_seconds for row in active
        ),
        "observed_interval_duration_at_least_threshold_fraction": (
            sum(row["observed_interval_seconds"] >= late_seconds for row in active) / len(active)
            if active
            else None
        ),
        "explicit_omission_cutoff_seconds": _distribution(
            [row["omission_cutoff_seconds"] for row in active]
        ),
        "no_omission_cutoff_games": sum(row["omission_cutoff_seconds"] is None for row in active),
        "late_threshold_seconds": late_seconds,
        "retained_horizon_at_least_threshold_games": late,
        "retained_horizon_at_least_threshold_fraction": late / len(active) if active else None,
        "observation_at_or_after_threshold_games": observed_late,
        "observation_at_or_after_threshold_fraction": (
            observed_late / len(active) if active else None
        ),
        "games_with_any_inferred_opening_deal": sum(row["inferred_opening"] for row in active),
        "source_requested_or_given_commands_used": sum(row["source_used"] for row in active),
        "verified_complete_matches": None,
    }


def audit_dataset(
    dataset,
    verify_shards=True,
    late_seconds=120.0,
    progress=None,
    *,
    a_horizon_seconds=60.0,
    a_min_labels=8,
    b_horizon_seconds=30.0,
    b_min_labels=4,
):
    manifest_path = Path(dataset)
    if manifest_path.is_dir():
        manifest_path /= "dataset.manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "crrl-bc-shards-v1":
        raise ValueError("Expected crrl-bc-shards-v1 dataset")
    metadata_path = manifest_path.parent / manifest["metadata"]
    metadata_bytes = metadata_path.read_bytes()
    if hashlib.sha256(metadata_bytes).hexdigest() != manifest["metadata_sha256"]:
        raise ValueError("Metadata checksum mismatch; producer may be publishing, retry")
    metadata = json.loads(metadata_bytes)
    replay_meta = metadata["replays"]
    count = len(replay_meta)
    excluded = set(metadata.get("excluded_replay_ids", []))
    if any(index < 0 or index >= count for index in excluded):
        raise ValueError("Excluded replay ID is outside metadata")
    decision_ticks = int(metadata["decision_ticks"])
    if decision_ticks < 1:
        raise ValueError("Invalid decision interval")
    fields = {
        name: np.zeros(count, np.int64)
        for name in (
            "observation_rows",
            "valid_rows",
            "valid_card_plays",
            "valid_abilities",
            "valid_waits",
            "flagged_rows",
            "unflagged_illegal_rows",
            "seat_bits",
            "unique_intervals",
            "unique_seat_intervals",
        )
    }
    assigned_ranges = np.zeros(count, np.uint8)
    first_tick = np.full(count, np.iinfo(np.int64).max, np.int64)
    last_tick = np.full(count, -1, np.int64)
    raw_rows = 0
    for number, shard in enumerate(manifest["shards"], 1):
        source = manifest_path.parent / shard["path"]
        left = int(shard["replay_offset"])
        right = left + int(shard["replays"])
        if left < 0 or right > count or np.any(assigned_ranges[left:right]):
            raise ValueError("Shard replay ranges overlap or are outside metadata")
        assigned_ranges[left:right] = 1
        if verify_shards and _sha256(source) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {source}")
        with np.load(source, allow_pickle=False) as arrays:
            # Deliberately never request arrays['observations'].
            actions = arrays["actions"].astype(np.int64)
            flags = arrays["flags"]
            ticks = arrays["ticks"].astype(np.int64)
            seats = arrays["seats"].astype(np.int64)
            local_ids = arrays["replay_ids"].astype(np.int64)
            ids = local_ids + int(shard["replay_offset"])
            n = len(actions)
            if any(len(values) != n for values in (flags, ticks, seats, ids)):
                raise ValueError(f"Mismatched row arrays: {source}")
            if n and (
                np.any(local_ids < 0)
                or np.any(local_ids >= int(shard["replays"]))
                or np.any(ids >= count)
                or np.any(ticks < 0)
                or np.any((seats < 0) | (seats > 1))
            ):
                raise ValueError(f"Invalid replay/tick/seat IDs: {source}")
            if n != int(shard["rows"]):
                raise ValueError(f"Shard row count differs from manifest: {source}")
            masks = arrays["masks"]
            if masks.ndim != 2 or len(masks) != n:
                raise ValueError(f"Invalid legal-mask shape: {source}")
            in_range = (actions >= 0) & (actions < masks.shape[1])
            legal = np.zeros(n, dtype=bool)
            selected = np.flatnonzero(in_range)
            legal[selected] = masks[selected, actions[selected]].astype(bool)
            valid = (flags == 0) & legal

            def add(name, selected_rows, shard_ids=ids):
                fields[name][:] += np.bincount(shard_ids[selected_rows], minlength=count)

            add("observation_rows", np.ones(n, bool))
            add("valid_rows", valid)
            add("valid_card_plays", valid & (actions > 0) & (actions < 2305))
            add("valid_abilities", valid & (actions >= 2305))
            add("valid_waits", valid & (actions == 0))
            add("flagged_rows", flags != 0)
            add("unflagged_illegal_rows", (flags == 0) & ~legal)
            np.minimum.at(first_tick, ids, ticks)
            np.maximum.at(last_tick, ids, ticks)
            np.bitwise_or.at(fields["seat_bits"], ids, 1 << seats)
            grid = np.unique(np.stack((ids, ticks), axis=-1), axis=0)
            fields["unique_intervals"] += np.bincount(grid[:, 0], minlength=count)
            grid = np.unique(np.stack((ids, ticks, seats), axis=-1), axis=0)
            fields["unique_seat_intervals"] += np.bincount(grid[:, 0], minlength=count)
            raw_rows += n
        if progress:
            progress(number, len(manifest["shards"]), raw_rows)
    if raw_rows != int(manifest["rows"]) or raw_rows != int(metadata["rows"]):
        raise ValueError("Total row count differs from manifest/metadata")
    if not assigned_ranges.all():
        raise ValueError("Metadata contains replay IDs outside all shard ranges")
    records = []
    for index, meta in enumerate(replay_meta):
        present = fields["observation_rows"][index] > 0
        cutoff = meta.get("supervision_cutoff_tick")
        row = {
            "replay_id": index,
            "identity": replay_identity(meta),
            "source": meta.get("source"),
            "replay_tag": meta["replay_tag"],
            "excluded": index in excluded,
            "inferred_opening": any(meta.get("opening_cycle_inference", [])),
            "initial_deal_policy": meta.get("initial_deal_policy", "unknown_or_seeded"),
            "source_used": int(meta.get("used", 0)),
            "omission_cutoff_seconds": float(cutoff) * 0.05 if cutoff is not None else None,
            "first_observation_seconds": float(first_tick[index]) * 0.05 if present else None,
            "last_observation_seconds": float(last_tick[index]) * 0.05 if present else None,
            "retained_horizon_seconds": (
                float(last_tick[index] + decision_ticks) * 0.05 if present else None
            ),
            "retained_span_seconds": (
                float(last_tick[index] - first_tick[index] + decision_ticks) * 0.05
                if present
                else None
            ),
            "observed_seats": int(fields["seat_bits"][index]).bit_count(),
            "observed_interval_seconds": float(fields["unique_intervals"][index])
            * decision_ticks
            * 0.05,
            "complete_paired_history_grid": bool(
                present
                and (last_tick[index] - first_tick[index]) % decision_ticks == 0
                and fields["unique_intervals"][index] * decision_ticks
                == last_tick[index] - first_tick[index] + decision_ticks
                and fields["unique_seat_intervals"][index]
                == fields["observation_rows"][index]
                == 2 * fields["unique_intervals"][index]
            ),
            **{key: int(values[index]) for key, values in fields.items() if key != "seat_bits"},
        }
        row["collisions"] = int(meta.get("collisions", 0))
        row["reconstruction_policy"] = meta.get("reconstruction_policy", "unknown")
        row["quality_tier"] = quality_tier(
            row,
            a_horizon_seconds=a_horizon_seconds,
            a_min_labels=a_min_labels,
            b_horizon_seconds=b_horizon_seconds,
            b_min_labels=b_min_labels,
        )
        records.append(row)
    identities = defaultdict(list)
    for row in records:
        identities[row["identity"]].append(row)
    active_duplicates = sum(
        sum(not row["excluded"] for row in group) > 1 for group in identities.values()
    )
    cycle_overrides = priority_violations = 0
    for group in identities.values():
        selected = [row for row in group if not row["excluded"]]
        rejected = [row for row in group if row["excluded"]]
        if len(selected) == 1 and selected[0]["inferred_opening"]:
            cycle_overrides += sum(not row["inferred_opening"] for row in rejected)
        if selected and not any(row["inferred_opening"] for row in selected):
            priority_violations += int(any(row["inferred_opening"] for row in rejected))
    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "metadata_sha256": manifest["metadata_sha256"],
        "shard_hashes_verified": verify_shards,
        "decision_ticks": decision_ticks,
        **_summary(
            records,
            late_seconds,
            a_horizon_seconds=a_horizon_seconds,
            a_min_labels=a_min_labels,
            b_horizon_seconds=b_horizon_seconds,
            b_min_labels=b_min_labels,
        ),
        "active_duplicate_identity_groups": active_duplicates,
        "excluded_seeded_duplicates_overridden_by_active_cycle": cycle_overrides,
        "active_seeded_with_excluded_inferred_cycle_identity_groups": priority_violations,
        "by_source": {
            str(source): _summary(
                [row for row in records if row["source"] == source],
                late_seconds,
                a_horizon_seconds=a_horizon_seconds,
                a_min_labels=a_min_labels,
                b_horizon_seconds=b_horizon_seconds,
                b_min_labels=b_min_labels,
            )
            for source in sorted({row["source"] for row in records}, key=str)
        },
    }
    return summary, records


def compare_coverage(before, after):
    def unique_index(records):
        groups = defaultdict(list)
        for row in records:
            if not row["excluded"]:
                groups[row["identity"]].append(row)
        return (
            {key: rows[0] for key, rows in groups.items() if len(rows) == 1},
            sum(len(rows) > 1 for rows in groups.values()),
        )

    previous, old_ambiguous = unique_index(before)
    current, new_ambiguous = unique_index(after)
    keys = sorted(previous.keys() & current.keys())
    replaced = [
        key
        for key in keys
        if current[key]["inferred_opening"] and not previous[key]["inferred_opening"]
    ]

    def paired(selected):
        old = [previous[key] for key in selected]
        new = [current[key] for key in selected]
        deltas = [
            b["retained_horizon_seconds"] - a["retained_horizon_seconds"]
            for a, b in zip(old, new, strict=True)
            if a["retained_horizon_seconds"] is not None
            and b["retained_horizon_seconds"] is not None
        ]
        return {
            "games": len(selected),
            "before_valid_card_play_labels": sum(row["valid_card_plays"] for row in old),
            "after_valid_card_play_labels": sum(row["valid_card_plays"] for row in new),
            "before_retained_horizon_seconds": _distribution(
                [row["retained_horizon_seconds"] for row in old]
            ),
            "after_retained_horizon_seconds": _distribution(
                [row["retained_horizon_seconds"] for row in new]
            ),
            "paired_retained_horizon_delta_seconds": _distribution(deltas),
        }

    return {
        "matched_unambiguous_identities": paired(keys),
        "seeded_to_inferred_cycle_replacements": paired(replaced),
        "new_unambiguous_identities": len(current.keys() - previous.keys()),
        "removed_unambiguous_identities": len(previous.keys() - current.keys()),
        "ambiguous_active_identity_groups_before": old_ambiguous,
        "ambiguous_active_identity_groups_after": new_ambiguous,
        "raw_source_used_is_not_actual_success_count": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--late-seconds", type=float, default=120.0)
    parser.add_argument("--a-horizon-seconds", type=float, default=60.0)
    parser.add_argument("--a-min-labels", type=int, default=8)
    parser.add_argument("--b-horizon-seconds", type=float, default=30.0)
    parser.add_argument("--b-min-labels", type=int, default=4)
    parser.add_argument("--no-verify-shards", action="store_true")
    args = parser.parse_args()
    if (
        not np.isfinite(args.late_seconds)
        or args.late_seconds < 0
        or not np.isfinite(args.a_horizon_seconds)
        or not np.isfinite(args.b_horizon_seconds)
        or args.a_horizon_seconds < args.b_horizon_seconds
        or args.b_horizon_seconds < 0
        or args.a_min_labels < args.b_min_labels
        or args.b_min_labels < 1
    ):
        parser.error("invalid quality thresholds")
    results, records = {}, {}
    for label, path in (("before", args.before), ("after", args.after)):

        def progress(done, total, rows, audit_label=label):
            if done % 16 == 0 or done == total:
                print(
                    json.dumps(
                        {
                            "audit": audit_label,
                            "shards_done": done,
                            "shards_total": total,
                            "rows_scanned": rows,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        results[label], records[label] = audit_dataset(
            path,
            not args.no_verify_shards,
            args.late_seconds,
            progress,
            a_horizon_seconds=args.a_horizon_seconds,
            a_min_labels=args.a_min_labels,
            b_horizon_seconds=args.b_horizon_seconds,
            b_min_labels=args.b_min_labels,
        )
    report = {
        "protocol": "retained observation intervals and exact legal unmasked BC labels",
        "scope": (
            "All active replay IDs in each supplied snapshot, not only its train split. "
            "Retained horizon=(last saved tick+decision_ticks)*0.05; end is uncertain within "
            "one decision interval. A missing omission cutoff does not establish a complete "
            "match, and longer coverage does not prove fidelity to original human states."
        ),
        **results,
        "comparison": compare_coverage(records["before"], records["after"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "report": str(args.out),
                "before": results["before"]["valid_card_play_labels"],
                "after": results["after"]["valid_card_play_labels"],
            }
        )
    )


if __name__ == "__main__":
    main()
