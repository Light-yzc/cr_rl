import json

import numpy as np
import pytest

from cr_rl.data.bc_provenance import dataset_kind, require_dataset_kind
from cr_rl.data.bc_snapshot import freeze_datasets
from cr_rl.training.persistence import file_sha256
from scripts.prepare_clean_bc_run import prepare_snapshot


def source_dataset(root, *, repaired=False):
    root.mkdir()
    # Two short complete grids, deliberately below the A-tier horizon/label
    # thresholds. Repair experiments may still train these with explicit opt-in.
    ticks = np.tile(np.repeat(np.arange(0, 20, 5), 2), 2)
    seats = np.tile([0, 1], 8)
    ids = np.repeat([0, 1], 8)
    actions = np.zeros(16, np.int32)
    actions[[0, 8]] = 1
    shard = root / "data.npz"
    np.savez_compressed(
        shard,
        observations=np.zeros((16, 3), np.int32),
        masks=np.ones((16, 2307), np.uint8),
        actions=actions,
        ticks=ticks,
        seats=seats,
        replay_ids=ids,
        flags=np.zeros(16, np.int32),
    )
    metadata = {
        "dataset_kind": "repaired" if repaired else "exact",
        "schema": {},
        "vocabulary": {},
        "backend_commit": "pinned",
        "content_sha": "tables",
        "level": 11,
        "decision_ticks": 5,
        "rows": len(actions),
        "replays": [
            {
                "source": "fixture",
                "replay_tag": f"match-{i}",
                "canonical_game_signature": f"match-{i}",
                "reconstruction_policy": "first_omission_prefix",
                **({"repair_is_exact_bc": False} if repaired else {}),
            }
            for i in range(2)
        ],
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    (root / "dataset.manifest.json").write_text(
        json.dumps(
            {
                "format": "crrl-bc-shards-v1",
                "metadata": metadata_path.name,
                "metadata_sha256": file_sha256(metadata_path),
                "rows": len(actions),
                "replays": 2,
                "shards": [
                    {
                        "path": shard.name,
                        "sha256": file_sha256(shard),
                        "replay_offset": 0,
                        "replays": 2,
                        "rows": len(actions),
                    }
                ],
            }
        )
    )
    return root


@pytest.mark.parametrize(
    "record",
    [
        {"import_policy": {"repair_is_exact_bc": False}},
        {"import_policy": {"repair_time": True}},
        {"replays": [{"repair_hand": True}]},
        {"replays": [{"repair_records": [{"category": "tile_snap"}]}]},
        {"replays": [{"repair_summary": {"delayed_events": 1}}]},
    ],
)
def test_legacy_repair_provenance_cannot_be_promoted_as_exact(record):
    assert dataset_kind(record) == "repaired"
    with pytest.raises(ValueError, match="requires explicit"):
        require_dataset_kind(record)
    assert require_dataset_kind(record, allow_repaired=True) == "repaired"
    with pytest.raises(ValueError, match="declares exact"):
        dataset_kind({**record, "dataset_kind": "exact"})


def test_freeze_requires_opt_in_and_never_silently_merges_dataset_kinds(tmp_path):
    exact = source_dataset(tmp_path / "exact")
    repair = source_dataset(tmp_path / "repair", repaired=True)
    with pytest.raises(ValueError, match="requires explicit"):
        freeze_datasets([(repair, repair)], tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    with pytest.raises(ValueError, match="separate BC snapshot"):
        freeze_datasets(
            [(exact, exact), (repair, repair)], tmp_path / "mixed", allow_repaired=True
        )
    assert not (tmp_path / "mixed").exists()
    out = tmp_path / "frozen"
    result = freeze_datasets([(repair, repair)], out, deduplicate=True, allow_repaired=True)
    assert result["dataset_kind"] == "repaired"
    assert json.loads((out / "dataset_metadata.json").read_text())["dataset_kind"] == "repaired"


def test_manifest_repair_marker_cannot_be_lost_in_legacy_metadata():
    with pytest.raises(ValueError, match="requires explicit"):
        require_dataset_kind({}, manifest={"dataset_kind": "repaired"})
    with pytest.raises(ValueError, match="declares exact"):
        require_dataset_kind(
            {"repair_is_exact_bc": False}, manifest={"dataset_kind": "exact"}
        )


def test_ordinary_execution_ledger_does_not_misclassify_exact_source_data():
    metadata = {
        "dataset_kind": "exact",
        "replays": [
            {
                "repair_records": [
                    {"category": "execution", "accepted": True},
                    {"category": "rejected", "reason": "not_in_hand"},
                ]
            }
        ],
    }
    assert dataset_kind(metadata) == "exact"
    metadata["replays"][0]["repair_records"].append({"category": "time_delay"})
    with pytest.raises(ValueError, match="declares exact"):
        dataset_kind(metadata)


def test_clean_preparation_rejects_repair_but_explicit_experiment_keeps_short_games(tmp_path):
    repair = source_dataset(tmp_path / "repair", repaired=True)
    thresholds = dict(a_horizon_seconds=60, a_min_labels=8, b_horizon_seconds=30, b_min_labels=4)
    with pytest.raises(ValueError, match="requires explicit"):
        prepare_snapshot(repair, tmp_path / "blocked", min_a_games=2, thresholds=thresholds)
    out = tmp_path / "experimental"
    summary, selected, excluded = prepare_snapshot(
        repair, out, min_a_games=2, thresholds=thresholds, allow_repaired=True
    )
    assert summary["quality_tiers"] == {"A": 0, "B": 0, "C": 2}
    assert (selected, excluded) == (2, 0)
    metadata = json.loads((out / "dataset_metadata.json").read_text())
    assert metadata["dataset_kind"] == "repaired"
    assert metadata["experimental_repair_snapshot"] is True
    assert metadata["clean_snapshot"] is False
    assert metadata["excluded_replay_ids"] == []
    report = json.loads((out / "quality_report.json").read_text())
    assert report["a_tier_replay_ids"] == []
    assert report["selected_replay_ids"] == [0, 1]
