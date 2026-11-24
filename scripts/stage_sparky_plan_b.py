"""Stage a balanced, game-disjoint Sparky and general BC snapshot on local SSD."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import random
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile


CARD = "ZapMachine"
PLAN = "sparky_b"
DEFAULT_SEED = 20261121
PARTITIONS = ("train", "validation")
COHORTS = ("sparky", "general")
MIN_PER_COHORT = 6


@dataclass(frozen=True)
class Game:
    identity: str
    decks: list
    source_path: str
    seats: tuple[int, ...]


def _path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe snapshot path {name!r}")
    return name


def _read_local(root, name):
    p = (root / _path(name)).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"File escaped snapshot directory {name!r}")
    return p.read_bytes()


def _metadata(read_bytes):
    manifest = json.loads(read_bytes("dataset.manifest.json"))
    if manifest.get("format") != "crrl-bc-shards-v1" or not manifest.get("snapshot_id"):
        raise ValueError("Expected a named CRRL BC shard snapshot")
    metadata = json.loads(read_bytes(_path(manifest["metadata"])))
    if metadata.get("decision_ticks") != 8:
        raise ValueError("Plan B requires native 8-tick replay history")
    splits = json.loads(read_bytes("split_registry.json"))["assignments"]
    return manifest, metadata, splits


def _index_rows(index):
    if index.get("format") != "crrl-sparky-bc-index-v1" or index.get("label_card") != CARD:
        raise ValueError("Expected the original Sparky index with whole-game splits")
    rows = {}
    for row in index["games"]:
        identity = row["identity"]
        if identity in rows or row["partition"] not in (*PARTITIONS, "heldout"):
            raise ValueError("Invalid original Sparky split record")
        rows[identity] = row
    return rows


def _games(manifest, metadata, splits, cohort, sparky_rows):
    by_partition = {partition: [] for partition in PARTITIONS}
    seen = set()
    offsets = set()
    paths = set()
    for shard in manifest["shards"]:
        offset = shard["replay_offset"]
        path = _path(shard["path"])
        if type(offset) is not int or not 0 <= offset < len(metadata["replays"]):
            raise ValueError("Invalid replay offset in BC shard")
        if offset in offsets or path in paths:
            raise ValueError("Duplicate replay offset or path in BC snapshot")
        offsets.add(offset)
        paths.add(path)
        record = metadata["replays"][offset]
        identity = record["canonical_game_signature"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Missing or duplicate canonical game identity")
        seen.add(identity)
        decks = record["decks"]
        if len(decks) != 2:
            raise ValueError(f"Missing two-seat deck data for {identity}")
        sparky_seats = {seat for seat, deck in enumerate(decks) if CARD in deck}
        candidates = tuple(shard.get("selected_seats", (0, 1)))
        if (not candidates or any(type(seat) is not int or seat not in (0, 1) for seat in candidates)
                or len(candidates) != len(set(candidates))):
            raise ValueError(f"Invalid selected seats for {identity}")
        partition = splits.get(identity)
        if partition not in (*PARTITIONS, "heldout"):
            raise ValueError(f"Invalid source partition for {identity}")
        if cohort == "sparky":
            original = sparky_rows[identity]
            if original["decks"] != decks or set(original["seats"]) != sparky_seats or set(candidates) != sparky_seats:
                raise ValueError(f"Sparky seat or deck provenance differs for {identity}")
            partition = original["partition"]
        elif identity in sparky_rows or sparky_seats:
            # Exclude the entire game, including either non-Sparky seat.
            continue
        if partition in PARTITIONS:
            by_partition[partition].append(Game(identity, decks, path, candidates))
    if len(offsets) != len(metadata["replays"]):
        raise ValueError("Manifest and replay metadata disagree")
    if cohort == "sparky" and seen != set(sparky_rows):
        raise ValueError("Sparky snapshot does not cover the original indexed games")
    return by_partition


def _select(pools, seed):
    # Choose 12.5% validation in both cohorts without moving any original split.
    pairs = min(
        len(pools[cohort][partition]) // multiple
        for cohort in COHORTS
        for partition, multiple in zip(PARTITIONS, (7, 1))
    )
    if pairs < MIN_PER_COHORT:
        raise ValueError("Plan B needs at least six original validation games per cohort and a 12.5% validation split")
    rng = random.Random(seed)
    selected = []
    for partition, multiple in zip(PARTITIONS, (7, 1)):
        sampled = {
            cohort: rng.sample(sorted(pools[cohort][partition], key=lambda g: g.identity), pairs * multiple)
            for cohort in COHORTS
        }
        for games in zip(*(sampled[cohort] for cohort in COHORTS)):
            for cohort, game in zip(COHORTS, games):
                selected.append((cohort, partition, game, rng.choice(game.seats)))
    return selected


def _write_snapshot(out, sparky, archive, metadata, selected, snapshots, seed):
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temp:
        staging = Path(temp) / "snapshot"
        (staging / "games").mkdir(parents=True)
        replays, shards, assignments = [], [], {}
        for i, (cohort, partition, game, seat) in enumerate(selected):
            relative = f"games/{i:06d}.npz"
            dest = staging / relative
            if cohort == "sparky":
                path = (sparky / game.source_path).resolve()
                if not path.is_relative_to(sparky) or not path.is_file():
                    raise ValueError(f"Missing or unsafe Sparky game {game.source_path}")
                try:
                    dest.hardlink_to(path)
                except OSError:
                    shutil.copyfile(path, dest)
                size = path.stat().st_size
            else:
                with archive.open(game.source_path) as origin, dest.open("xb") as target:
                    shutil.copyfileobj(origin, target)
                size = archive.getinfo(game.source_path).file_size
            if size <= 0 or size != dest.stat().st_size:
                raise OSError(f"Incomplete BC shard {game.source_path}")
            if game.identity in assignments:
                raise ValueError(f"Game appeared in two Plan B groups {game.identity}")
            assignments[game.identity] = partition
            replays.append(dict(canonical_game_signature=game.identity, decks=game.decks, cohort=cohort))
            shards.append(dict(path=relative, replay_offset=i, selected_seats=[seat], cohort=cohort))
        snapshot = f"sparky_plan_b_{uuid4().hex}"
        documents = {
            "dataset_metadata.json": dict(
                schema=metadata["schema"], vocabulary=metadata["vocabulary"], decision_ticks=8,
                training_ready=False, expert_skill_verified=False, bc_plan=PLAN, validation_enabled=True,
                source_snapshots=snapshots, seed=seed, replays=replays,
            ),
            "dataset.manifest.json": dict(
                snapshot_id=snapshot, metadata="dataset_metadata.json", format="crrl-bc-shards-v1",
                shards=shards, decision_ticks=8,
            ),
            "split_registry.json": dict(snapshot_id=snapshot, assignments=assignments),
        }
        for name, data in documents.items():
            (staging / name).write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        staging.rename(out)
    counts = Counter((cohort, partition) for cohort, partition, _, _ in selected)
    return dict(
        snapshot_id=snapshot, output=str(out), source_snapshots=snapshots,
        games=len(selected), validation_fraction=sum(assignments[x] == "validation" for x in assignments) / len(assignments),
        per_cohort={cohort: {p: counts[cohort, p] for p in PARTITIONS} for cohort in COHORTS},
    )


def stage(sparky, index_path, general_archive, out, *, seed=DEFAULT_SEED):
    sparky, out = sparky.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(Path("/content/drive")):
        raise ValueError("Plan B requires an unused output directory on the runtime's local SSD")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    sparky_manifest, sparky_meta, sparky_splits = _metadata(lambda name: _read_local(sparky, name))
    if sparky_manifest["snapshot_id"] != index["snapshot_id"] + "_all_train":
        raise ValueError("Sparky snapshot and original split index differ")
    if any(index["metadata"].get(k) != sparky_meta[k] for k in ("schema", "vocabulary", "decision_ticks")):
        raise ValueError("Sparky snapshot schema or vocabulary differs from its index")
    sparky_rows = _index_rows(index)
    with ZipFile(general_archive) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate file names in general quality BC archive")
        general_manifest, general_meta, general_splits = _metadata(archive.read)
        for k in ("schema", "vocabulary"):
            if sparky_meta[k] != general_meta[k]:
                raise ValueError(f"Sparky and general BC {k} differ")
        pools = {
            "sparky": _games(sparky_manifest, sparky_meta, sparky_splits, "sparky", sparky_rows),
            "general": _games(general_manifest, general_meta, general_splits, "general", sparky_rows),
        }
        selected = _select(pools, seed)
        return _write_snapshot(
            out, sparky, archive, sparky_meta, selected,
            {"sparky": sparky_manifest["snapshot_id"], "general": general_manifest["snapshot_id"]}, seed,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("sparky", "index", "general-archive", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    archive = args.general_archive.resolve()
    if archive.is_relative_to(Path("/content/drive")):
        parser.error("Copy the general archive to local SSD once before staging")
    print(json.dumps(stage(args.sparky, args.index, archive, args.out, seed=args.seed), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
