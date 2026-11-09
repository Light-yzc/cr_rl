import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_bc_coverage import audit_dataset, compare_coverage, quality_tier, replay_identity


def fixture(root, replays, ids, ticks, actions, flags, seats, excluded=(), illegal=()):
    root.mkdir()
    shard = root / "data.npz"
    masks = np.ones((len(actions), 2307), np.uint8)
    for row in illegal:
        masks[row, actions[row]] = 0
    np.savez_compressed(
        shard,
        actions=np.array(actions, np.int32),
        replay_ids=np.array(ids, np.int32),
        ticks=np.array(ticks, np.int32),
        flags=np.array(flags, np.int32),
        seats=np.array(seats, np.int32),
        masks=masks,
    )
    # Audit must not parse or decompress observation tensors at all.
    with zipfile.ZipFile(shard, "a") as archive:
        archive.writestr("observations.npy", b"deliberately invalid observation member")
    metadata = root / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "replays": replays,
                "rows": len(actions),
                "decision_ticks": 5,
                "excluded_replay_ids": list(excluded),
            }
        )
    )
    manifest = {
        "format": "crrl-bc-shards-v1",
        "rows": len(actions),
        "metadata": metadata.name,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "shards": [
            {
                "path": shard.name,
                "rows": len(actions),
                "replay_offset": 0,
                "replays": len(replays),
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "dataset.manifest.json").write_text(json.dumps(manifest))
    return root


class BCCoverageTest(unittest.TestCase):
    def test_quality_tier_fails_closed_without_reconstruction_provenance(self):
        self.assertEqual(
            quality_tier(
                {
                    "excluded": False,
                    "retained_horizon_seconds": 90,
                    "valid_card_plays": 8,
                    "valid_abilities": 0,
                    "complete_paired_history_grid": True,
                    "observed_seats": 2,
                    "collisions": 0,
                }
            ),
            "C",
        )

    def test_exact_valid_labels_exclusion_and_cutoff_none_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture(
                Path(directory) / "snapshot",
                [
                    {
                        "source": "new",
                        "replay_tag": "new",
                        "canonical_game_signature": "same",
                        "opening_cycle_inference": [{"order": list(range(8))}],
                        "supervision_cutoff_tick": 2600,
                        "used": 99,
                    },
                    {
                        "source": "old",
                        "replay_tag": "old",
                        "canonical_game_signature": "same",
                        "supervision_cutoff_tick": 400,
                        "used": 2,
                    },
                    {"source": "new", "replay_tag": "short", "used": 80},
                ],
                [0, 0, 0, 1, 1, 2, 2],
                [0, 2395, 2400, 0, 5, 0, 195],
                [1, 0, 2, 3, 4, 3, 2305],
                [0, 0, 65536, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1],
                excluded=[1],
                illegal=[5],
            )
            summary, records = audit_dataset(root)
            self.assertEqual(summary["active_metadata_games"], 2)
            self.assertEqual(summary["valid_card_play_labels"], 1)
            self.assertEqual(summary["valid_ability_labels"], 1)
            self.assertEqual(summary["valid_wait_labels"], 1)
            self.assertEqual(summary["unflagged_illegal_target_rows"], 1)
            self.assertEqual(summary["source_requested_or_given_commands_used"], 179)
            self.assertEqual(summary["explicit_omission_cutoff_seconds"]["median"], 130)
            self.assertEqual(summary["no_omission_cutoff_games"], 1)
            self.assertIsNone(summary["verified_complete_matches"])
            self.assertEqual(records[0]["retained_horizon_seconds"], 120.25)
            self.assertEqual(records[2]["retained_horizon_seconds"], 10)
            self.assertEqual(summary["retained_horizon_at_least_threshold_fraction"], 0.5)
            self.assertEqual(summary["observed_interval_duration_at_least_threshold_fraction"], 0)
            self.assertEqual(summary["active_games_with_complete_paired_history_grid"], 0)
            self.assertEqual(summary["excluded_seeded_duplicates_overridden_by_active_cycle"], 1)

    def test_real_paired_grid_duration_and_matched_cycle_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ticks = np.repeat(np.arange(0, 2405, 5), 2)
            source = {"source": "IL", "source_revision": "pinned", "replay_tag": "game"}
            before = fixture(
                root / "before",
                [source],
                [0] * 4,
                [0, 0, 5, 5],
                [1, 0, 0, 0],
                [0] * 4,
                [0, 1, 0, 1],
            )
            after = fixture(
                root / "after",
                [{**source, "opening_cycle_inference": [True]}],
                [0] * len(ticks),
                ticks,
                [1] + [0] * (len(ticks) - 1),
                [0] * len(ticks),
                np.tile([0, 1], len(ticks) // 2),
            )
            _, old = audit_dataset(before)
            summary, new = audit_dataset(after)
            self.assertEqual(summary["active_games_with_complete_paired_history_grid"], 1)
            self.assertEqual(summary["observed_interval_duration_at_least_threshold_fraction"], 1)
            self.assertEqual(new[0]["observed_interval_seconds"], 120.25)
            compared = compare_coverage(old, new)
            self.assertEqual(compared["seeded_to_inferred_cycle_replacements"]["games"], 1)
            self.assertEqual(compared["matched_unambiguous_identities"]["games"], 1)
            self.assertEqual(replay_identity(source), json.dumps(["IL", "pinned", "game"]))
            changed_revision = {**source, "source_revision": "different"}
            self.assertNotEqual(replay_identity(source), replay_identity(changed_revision))

    def test_checksum_rejects_changed_shard_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture(
                Path(directory) / "snapshot",
                [{"replay_tag": "x"}],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1],
            )
            with (root / "data.npz").open("ab") as stream:
                stream.write(b"changed archive")
            with self.assertRaisesRegex(ValueError, "Shard checksum"):
                audit_dataset(root)
            audit_dataset(root, verify_shards=False)
            (root / "metadata.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Metadata checksum"):
                audit_dataset(root)


if __name__ == "__main__":
    unittest.main()
