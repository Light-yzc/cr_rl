"""Verify the Plan B snapshot against its two source manifests and registries."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


spec = importlib.util.spec_from_file_location(
    "stage_sparky_plan_b", Path(__file__).resolve().parents[1] / "scripts/stage_sparky_plan_b.py"
)
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)

SPARKY_DECK = [staging.CARD] * 8
GENERAL_DECK = ["Archers"] * 8


def _write_dataset(path, cohort, rows, *, mismatch=False):
    metadata = dict(
        schema={"scales": {"elixir_scale": 10000 if not mismatch else 1000}},
        vocabulary={"names": [staging.CARD, "Archers"]},
        decision_ticks=8,
        replays=[],
    )
    manifest = dict(
        snapshot_id="focused_all_train" if cohort == "sparky" else "general_52",
        metadata="dataset_metadata.json",
        format="crrl-bc-shards-v1",
        shards=[],
    )
    index = dict(
        format="crrl-sparky-bc-index-v1", label_card=staging.CARD, snapshot_id="focused",
        metadata={k: metadata[k] for k in ("schema", "vocabulary", "decision_ticks")},
        games=[],
    )
    assignments = {}
    files = {}
    for i, (identity, partition, decks) in enumerate(rows):
        relative = f"games/{i:04}.npz"
        selected = [s for s, deck in enumerate(decks) if staging.CARD in deck]
        manifest["shards"].append(dict(path=relative, replay_offset=i, selected_seats=selected or [0, 1]))
        metadata["replays"].append(dict(canonical_game_signature=identity, decks=decks))
        index["games"].append(dict(identity=identity, partition=partition, decks=decks, seats=selected))
        assignments[identity] = "train" if cohort == "sparky" else partition
        files[relative] = identity.encode()
    files.update({
        "dataset.manifest.json": json.dumps(manifest).encode(),
        "split_registry.json": json.dumps(dict(assignments=assignments)).encode(),
        "dataset_metadata.json": json.dumps(metadata).encode(),
    })
    if cohort == "sparky":
        (path / "games").mkdir(parents=True)
        for name, payload in files.items():
            (path / name).write_bytes(payload)
        index_path = path.parent / "index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        return index_path
    with ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return path


def _sources(tmp_path, *, mismatch=False):
    # The extra train games and heldout games exercise the source partition constraints.
    focus = [(f"focused-{p}-{i}", p, [SPARKY_DECK, GENERAL_DECK]) for p, n in (
        ("train", 58), ("validation", 6), ("heldout", 8)
    ) for i in range(n)]
    focus[0] = (*focus[0][:2], [SPARKY_DECK, SPARKY_DECK])
    general = [(f"general-{p}-{i}", p, [GENERAL_DECK, GENERAL_DECK]) for p, n in (
        ("train", 77), ("validation", 13), ("heldout", 8)
    ) for i in range(n)]
    # The quality archive can include the same games and other Sparky games.
    general += focus[:20] + [("wrong-cohort", "validation", [GENERAL_DECK, SPARKY_DECK])]
    sparky = tmp_path / "focused"
    index = _write_dataset(sparky, "sparky", focus)
    archive = _write_dataset(tmp_path / "general.zip", "general", general, mismatch=mismatch)
    return sparky, index, archive, {x[0]: x[1] for x in focus + general}


def test_balanced_and_disjoint_plan_b(tmp_path):
    sparky, index, archive, original = _sources(tmp_path)
    out = tmp_path / "mixed"
    summary = staging.stage(sparky, index, archive, out)
    assert summary["games"] == 96 and summary["validation_fraction"] == 0.125
    assert summary["per_cohort"] == {
        name: {"train": 42, "validation": 6} for name in ("sparky", "general")
    }
    manifest = json.loads((out / "dataset.manifest.json").read_bytes())
    metadata = json.loads((out / manifest["metadata"]).read_bytes())
    splits = json.loads((out / "split_registry.json").read_bytes())["assignments"]
    assert metadata["validation_enabled"] and metadata["bc_plan"] == "sparky_b"
    assert len(splits) == len(metadata["replays"]) == len(manifest["shards"]) == 96
    assert splits == {identity: original[identity] for identity in splits}
    assert Counter(splits.values()) == {"train": 84, "validation": 12}
    for shard in manifest["shards"]:
        record = metadata["replays"][shard["replay_offset"]]
        assert record["cohort"] == shard["cohort"]
        (seat,) = shard["selected_seats"]
        assert (staging.CARD in record["decks"][seat]) == (record["cohort"] == "sparky")
        assert (out / shard["path"]).read_bytes() == record["canonical_game_signature"].encode()
    assert "wrong-cohort" not in splits


def test_mismatched_schema_is_rejected_before_publishing(tmp_path):
    sparky, index, archive, _ = _sources(tmp_path, mismatch=True)
    out = tmp_path / "mixed"
    with pytest.raises(ValueError, match="BC schema differ"):
        staging.stage(sparky, index, archive, out)
    assert not out.exists()
