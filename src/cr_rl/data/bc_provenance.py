"""Separate unmodified replay reconstruction from explicit repair experiments.

``exact`` means no runtime repair was enabled/applied.  It does not establish
fidelity to the human client's original state.
"""

from __future__ import annotations


def _repair_evidence(record):
    if record.get("repair_is_exact_bc") is False:
        return True
    if record.get("repair_class") in ("repaired", "mixed"):
        return True
    if any(record.get(flag) for flag in ("repair_hand", "repair_tile", "repair_time")):
        return True
    # The live executor's audit ledger also contains ordinary executions and
    # rejections. Those are evidence records, not state/action interventions.
    if any(
        item.get("category") not in {"execution", "rejected", "exact", "submitted", "native_run"}
        for item in record.get("repair_records", [])
    ):
        return True
    summary = record.get("repair_summary", {})
    return any(
        summary.get(key, 0)
        for key in (
            "repaired",
            "hand_replacements",
            "tile_snaps",
            "time_snaps",
            "delayed_events",
        )
    )


def dataset_kind(metadata):
    """Classify legacy/current manifests, rejecting a false exact declaration."""
    declared = metadata.get("dataset_kind")
    if declared not in (None, "exact", "repaired"):
        raise ValueError(f"Unsupported BC dataset_kind: {declared!r}")
    evidence = any(
        _repair_evidence(record)
        for record in (metadata, metadata.get("import_policy", {}), *metadata.get("replays", []))
    )
    if declared == "exact" and evidence:
        raise ValueError("BC dataset declares exact but contains runtime repair provenance")
    return "repaired" if declared == "repaired" or evidence else "exact"


def require_dataset_kind(metadata, *, allow_repaired=False, manifest=None):
    kind = dataset_kind(metadata)
    declared = (manifest or {}).get("dataset_kind")
    if declared not in (None, "exact", "repaired"):
        raise ValueError(f"Unsupported BC manifest dataset_kind: {declared!r}")
    if declared == "exact" and kind == "repaired":
        raise ValueError("BC manifest declares exact but metadata contains repair provenance")
    if declared == "repaired":
        if metadata.get("dataset_kind") == "exact":
            raise ValueError("BC manifest and metadata dataset_kind disagree")
        kind = "repaired"
    if kind == "repaired" and not allow_repaired:
        raise ValueError(
            "Repaired BC data requires explicit allow_repaired=True / --allow-repaired "
            "and a separate experimental snapshot"
        )
    return kind
