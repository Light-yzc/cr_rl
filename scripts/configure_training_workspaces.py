"""Publish useful BC and PPO charts while keeping counters and obsolete BC v2 hidden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb_workspaces.reports.v2 as panels
import wandb_workspaces.workspaces as workspaces

from cr_rl.training.persistence import atomic_json, copy_verified
from cr_rl.training.telemetry import create_training_workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--ppo-run", type=Path, required=True)
    args = parser.parse_args()
    tracked = json.loads((args.ppo_run / "wandb_run.json").read_text())
    entity, project = tracked["entity"], tracked["project"]
    ppo_url = create_training_workspace(entity, project, args.ppo_run.name)
    groups = [
        (
            "BC optimization",
            "bc/optimizer_steps",
            [
                ("Training loss", ["bc/train_loss"]),
                ("Gradient norm before clipping", ["bc/grad_norm"]),
            ],
        ),
        (
            "Held-out timing and card choice (metric v3)",
            "bc_v3/epoch",
            [
                ("Card accuracy conditional on playing", ["bc_v3/play_slot_accuracy"]),
                ("Play recall and WAIT accuracy", ["bc_v3/play_recall", "bc_v3/wait_accuracy"]),
                ("Balanced play / WAIT accuracy", ["bc_v3/balanced_wait_play_accuracy"]),
            ],
        ),
        (
            "Held-out placement (metric v3)",
            "bc_v3/epoch",
            [
                (
                    "Exact tile and within two tiles",
                    ["bc_v3/play_tile_accuracy", "bc_v3/play_tile_within_2"],
                ),
                (
                    "Mean tile Manhattan distance (lower is better)",
                    ["bc_v3/play_tile_mean_l1_distance"],
                ),
            ],
        ),
    ]
    bc_workspace = workspaces.Workspace(
        entity=entity,
        project=project,
        name="CR RL 119M BC corrected validation",
        auto_generate_panels=False,
        runset_settings=workspaces.RunsetSettings(query=args.bc_run.name + "_bc"),
        sections=[
            workspaces.Section(
                name=name,
                is_open=True,
                panels=[
                    panels.LinePlot(title=title, x=axis, y=metrics) for title, metrics in specs
                ],
            )
            for name, axis, specs in groups
        ],
    )
    bc_workspace.save()
    for folder, url in ((args.bc_run, bc_workspace.url), (args.ppo_run, ppo_url)):
        stable = f"https://wandb.ai/{entity}/{project}?nw=" + url.split("?nw=")[-1]
        metadata = {"url": stable, "sdk_url": url}
        atomic_json(folder / "wandb_workspace.json", metadata)
        copy_verified(
            folder / "wandb_workspace.json",
            Path("/content/drive/MyDrive/cr_rl/runs") / folder.name / "wandb_workspace.json",
        )
        print(json.dumps({"run": folder.name, **metadata}), flush=True)


if __name__ == "__main__":
    main()
