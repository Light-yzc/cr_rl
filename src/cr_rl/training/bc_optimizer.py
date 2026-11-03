"""Separate placement learning rates while retaining BC optimizer moments by name."""

from __future__ import annotations

import torch


def is_placement_parameter(name):
    return name in {"tile_queries", "slot_embedding"} or name.startswith(
        ("tile_context.", "tile_cross_attention.", "tile_actor.")
    )


def make_bc_optimizer(model, learning_rate, tile_learning_rate=None, fused=False):
    parameters = model.parameters()
    if tile_learning_rate is not None:
        parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not is_placement_parameter(n)],
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if is_placement_parameter(n)],
                "lr": tile_learning_rate,
            },
        ]
    return torch.optim.AdamW(parameters, lr=learning_rate, fused=fused)


def optimizer_parameter_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def carry_bc_optimizer(model, optimizer, saved, saved_names=None):
    if saved_names is None:
        if len(saved["param_groups"]) != 1:
            raise ValueError(
                "Legacy optimizer migration requires a single original parameter group"
            )
        saved_names = [[name for name, _ in model.named_parameters()]]
    if len(saved_names) != len(saved["param_groups"]):
        raise ValueError("Optimizer parameter-name groups disagree")
    by_name = {}
    groups_by_name = {}
    for group, names in zip(saved["param_groups"], saved_names, strict=True):
        if len(group["params"]) != len(names):
            raise ValueError("Optimizer parameter-name count differs")
        for index, name in zip(group["params"], names, strict=True):
            by_name[name] = saved["state"].get(index)
            groups_by_name[name] = group
    target = optimizer.state_dict()
    target_names = optimizer_parameter_names(model, optimizer)
    if set(by_name) != {name for names in target_names for name in names}:
        raise ValueError("Optimizer migration cannot change the model parameter set")
    target["state"] = {}
    for group, names in zip(target["param_groups"], target_names, strict=True):
        for key in (
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
        ):
            values = [groups_by_name[name][key] for name in names if key in groups_by_name[name]]
            if values:
                if any(value != values[0] for value in values):
                    raise ValueError(f"Cannot merge optimizer groups with different {key}")
                group[key] = values[0]
        for index, name in zip(group["params"], names, strict=True):
            if by_name[name] is not None:
                target["state"][index] = by_name[name]
    optimizer.load_state_dict(target)
