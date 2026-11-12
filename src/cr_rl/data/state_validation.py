"""Small, reproducible validation budgets; all inputs are local collection metadata."""

from __future__ import annotations

import json
import random
from pathlib import Path


def identity_keys(game):
    identity = game.get("canonical_game_signature") or game.get("game_signature")
    fallback = [game.get("source"), game.get("source_revision"), game.get("replay_tag")]
    key = str(identity) if identity else json.dumps(fallback)
    keys = {key}
    for field in ("canonical_game_signature", "game_signature"):
        if game.get(field):
            keys.add(str(game[field]))
    if game.get("replay_tag"):
        keys.update((json.dumps(fallback), json.dumps(fallback, separators=(",", ":"))))
    return key, keys


def resplit_validation_registry(games, registry, *, fraction=0.04, seed=20261111):
    """Shrink the prior validation pool; preserve all heldout identities."""
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be in (0,1)")
    original = registry.get("assignments", registry)
    records = {}
    for game in games:
        identity, keys = identity_keys(game)
        values = {original[key] for key in keys if key in original}
        if len(values) > 1:
            raise ValueError("Conflicting game split aliases")
        partition = next(iter(values)) if values else "train"
        if identity in records:
            raise ValueError("Duplicate validation pool identity")
        records[identity] = (keys, partition)
    eligible = sorted(key for key, (_, part) in records.items() if part != "heldout")
    target = min(len(eligible), max(1, round(len(eligible) * fraction)))
    prior_validation = sorted(key for key in eligible if records[key][1] == "validation")
    rng = random.Random(seed)
    chosen = set(rng.sample(prior_validation, min(target, len(prior_validation))))
    if len(chosen) < target:
        chosen.update(
            rng.sample([key for key in eligible if key not in chosen], target - len(chosen))
        )
    assignments = dict(original)
    counts = {"train": 0, "validation": 0, "heldout": 0}
    for identity, (keys, previous) in records.items():
        partition = (
            "heldout"
            if previous == "heldout"
            else ("validation" if identity in chosen else "train")
        )
        counts[partition] += 1
        for key in keys:
            assignments[key] = partition
    return {
        "assignments": assignments,
        "split_counts": counts,
        "validation_fraction": fraction,
        "validation_seed": seed,
        "validation_identities": sorted(chosen),
    }


def prepare_validation_registry(collection_root, destination, *, fraction=0.04, seed=20261111):
    """Create a small override registry without editing or repacking the dataset."""
    root, destination = Path(collection_root), Path(destination)
    collection = json.loads((root / "collection.manifest.json").read_text())
    version = {
        "source": collection.get("source_manifest_sha256"),
        "rows": collection["rows"],
        "games": collection["accepted_replays"],
        "fraction": fraction,
        "seed": seed,
    }
    if destination.is_file():
        cached = json.loads(destination.read_text())
        if cached.get("validation_budget_version") == version:
            return destination
    original = json.loads((root / "split_registry.json").read_text())
    games = []
    for batch in collection["batches"]:
        manifest_path = root / batch["manifest"]
        manifest = json.loads(manifest_path.read_text())
        metadata = json.loads((manifest_path.parent / manifest["metadata"]).read_text())
        games.extend(metadata["replays"])
    result = resplit_validation_registry(games, original, fraction=fraction, seed=seed)
    result["validation_budget_version"] = version
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    return destination


def sample_validation_games(assignments, limit, *, seed, epoch, optimizer_steps):
    """Sample whole games without touching the training RNG or crossing partitions."""
    eligible = sorted(replay_id for replay_id, part in assignments.items() if part == "validation")
    if not limit:
        return eligible
    rng = random.Random(f"validation:{seed}:{epoch}:{optimizer_steps}")
    return rng.sample(eligible, min(limit, len(eligible)))
