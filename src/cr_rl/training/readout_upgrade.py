"""Explicit, additive migration; never silently relax checkpoint loading."""

READOUT_PREFIXES = (
    "frame.spatial_projection.",
    "ordered_card_context.",
    "tile_spatial_",
    "tile_film.",
)


def load_upgraded_readouts(model, weights):
    if model.cfg.readout_version != 2:
        raise ValueError("Explicit readout-v2 target required")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(not key.startswith(READOUT_PREFIXES) for key in missing):
        raise ValueError(f"Unexpected migration keys: missing={missing}, unexpected={unexpected}")
    if not missing:
        raise ValueError("Source already contains upgraded readouts; use normal loading")
    return {"new_tensors": missing, "old_tensors_preserved": len(weights)}
