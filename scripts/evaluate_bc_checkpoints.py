"""Select BC epochs using every legal play label in the held-out replays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from train_bc import make_schema

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.bc import hierarchical_action_predictions
from cr_rl.training.persistence import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.dataset / "human_bc_reconstructed.npz"
    metadata = json.loads(source.with_suffix(".json").read_text())
    with np.load(source) as data:
        obs, masks, actions = data["observations"], data["masks"], data["actions"].astype(np.int64)
        replay_ids, seats, ticks, flags = (
            data[key] for key in ("replay_ids", "seats", "ticks", "flags")
        )
    schema = make_schema(metadata)
    adapter = CrForgeObservationAdapter(schema, len(metadata["vocabulary"]["names"]))
    last = torch.load(args.run_dir / "bc.pt", map_location="cpu", weights_only=False)
    if file_sha256(source) != last["dataset_sha256"]:
        raise ValueError("BC evaluation dataset differs from training")
    valid = (flags == 0) & masks[np.arange(len(actions)), actions].astype(bool)
    valid &= np.isin(replay_ids, last["validation_replays"])
    play_rows = np.flatnonzero(valid & (actions > 0) & (actions < 2305))
    wait_rows = np.flatnonzero(valid & (actions == 0))
    wait_rows = np.random.default_rng(20261103).choice(
        wait_rows, min(1024, len(wait_rows)), replace=False
    )
    rows = np.concatenate((play_rows, wait_rows))
    cfg = ModelConfig(**last["config"])
    epochs = last["epochs"]
    del last
    groups = {}
    for i in range(len(actions)):
        groups.setdefault((int(replay_ids[i]), int(seats[i])), []).append(i)
    positions = {}
    for key, indices in groups.items():
        indices.sort(key=lambda i: int(ticks[i]))
        positions[key] = {row: i for i, row in enumerate(indices)}
    torch.manual_seed(20261102)
    model = ActorCritic(cfg).cuda().eval()

    def evaluate():
        play_correct, wait_correct, tile_correct, play_recalled = 0, 0, 0, 0
        distances = []
        with torch.inference_mode():
            for start in range(0, len(rows), 4):
                ids = rows[start : start + 4]
                windows = np.zeros((len(ids), cfg.context_len, schema.seat_ints), np.int32)
                tm = np.zeros((len(ids), cfg.context_len), bool)
                for j, row in enumerate(ids):
                    key = (int(replay_ids[row]), int(seats[row]))
                    p = positions[key][int(row)]
                    seq = groups[key][max(0, p - cfg.context_len + 1) : p + 1]
                    windows[j, -len(seq) :] = obs[seq]
                    tm[j, -len(seq) :] = True
                h, e, em = adapter(windows)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
                ):
                    logp, _ = model(
                        *(
                            torch.as_tensor(x, device="cuda")
                            for x in (h, e, em, masks[ids].astype(bool), tm)
                        ),
                        **{
                            name: torch.as_tensor(
                                obs[ids][:, list(schema.header[field])], device="cuda"
                            )
                            for name, field in (
                                ("hand_card_ids", "hand_card"),
                                ("deck_card_ids", "deck"),
                                ("enemy_seen_card_ids", "enemy_seen"),
                            )
                        },
                    )
                logp = logp.float()
                placements = logp[:, 1:2305].reshape(len(ids), 4, 576)
                _, predicts_play_tensor, predictions_tensor = hierarchical_action_predictions(logp)
                predicts_play = predicts_play_tensor.cpu().numpy()
                predictions = predictions_tensor.cpu().numpy()
                for j, row in enumerate(ids):
                    action = int(actions[row])
                    if action == 0:
                        wait_correct += int(not predicts_play[j])
                    else:
                        slot, tile = divmod(action - 1, 576)
                        play_correct += int(predictions[j] == slot)
                        play_recalled += int(predicts_play[j])
                        predicted_tile = int(placements[j, slot].argmax().cpu())
                        tile_correct += int(predicted_tile == tile)
                        distances.append(
                            abs(predicted_tile % 18 - tile % 18)
                            + abs(predicted_tile // 18 - tile // 18)
                        )
        play_accuracy = play_correct / max(1, len(play_rows))
        wait_accuracy = wait_correct / max(1, len(wait_rows))
        return {
            "play_slot_accuracy": play_accuracy,
            "wait_accuracy": wait_accuracy,
            "balanced_wait_play_accuracy": (play_recalled / max(1, len(play_rows)) + wait_accuracy)
            / 2,
            "play_recall": play_recalled / max(1, len(play_rows)),
            "metric_version": 3,
            "play_tile_accuracy": tile_correct / max(1, len(play_rows)),
            "play_tile_mean_l1_distance": float(np.mean(distances)),
            "play_tile_within_2": float(np.mean(np.asarray(distances) <= 2)),
        }

    baseline = evaluate()
    tracking = None
    wandb_path = args.run_dir / "bc.wandb.json"
    if os.environ.get("WANDB_API_KEY") and wandb_path.exists():
        import wandb

        tracked = json.loads(wandb_path.read_text())
        prefix = tracked["url"].split("/runs/")[0].split("/")
        tracking = wandb.init(
            project=prefix[-1], entity=prefix[-2], id=tracked["id"], resume="allow"
        )
        tracking.define_metric("bc_v3/epoch", hidden=True)
        tracking.define_metric("bc_v3/*", step_metric="bc_v3/epoch")
        tracking.define_metric("bc/validation_*", hidden=True)
        tracking.summary["bc_metric_version"] = 3
        tracking.summary["bc_v3_baseline"] = baseline
        tracking.summary["bc_v3_validation_labels"] = {
            "plays": len(play_rows),
            "waits": len(wait_rows),
        }
        tracking.summary["superseded_metrics"] = (
            "bc/validation_* version 2 compared WAIT with individual slots. Use bc_v3/* instead."
        )
    reports = []
    for epoch in range(1, epochs + 1):
        state = torch.load(
            args.backup_dir / f"bc_epoch_{epoch:04d}.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(state["model"], strict=True)
        del state
        row = {"epoch": epoch, "validation": evaluate()}
        reports.append(row)
        print(json.dumps(row), flush=True)
        if tracking:
            tracking.log(
                {
                    "bc_v3/epoch": epoch,
                    **{f"bc_v3/{key}": value for key, value in row["validation"].items()},
                }
            )
    report = {
        "baseline": baseline,
        "epochs": reports,
        "play_labels": len(play_rows),
        "wait_labels": len(wait_rows),
        "protocol": "all held-out play labels + fixed WAIT sample",
    }
    atomic_json(args.run_dir / "bc_selection_full.json", report)
    if tracking:
        tracking.finish()


if __name__ == "__main__":
    main()
