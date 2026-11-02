from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models import ActorCritic, ModelConfig
from cr_rl.training.bc import hierarchical_bc_loss


def make_schema(metadata: dict):
    raw = metadata["schema"]
    header = {}
    for i, name in enumerate(raw["header_fields"]):
        if name != "reserved":
            header.setdefault(name, []).append(i)
    return SimpleNamespace(
        seat_ints=int(raw["seat_ints"]),
        header_ints=int(raw["header_ints"]),
        entity_ints=int(raw["entity_ints"]),
        max_entities=int(raw["max_entities"]),
        header={key: tuple(value) for key, value in header.items()},
        header_fields=tuple(raw["header_fields"]),
        entity_fields=tuple(raw["entity_fields"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--model",
        choices=("tiny", "small", "compact", "general", "large", "production"),
        default="tiny",
    )
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--play-fraction", type=float, default=0.5)
    parser.add_argument("--position-coef", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("artifacts/bc_tiny.pt"))
    args = parser.parse_args()
    if not 0.0 < args.play_fraction < 1.0:
        parser.error("--play-fraction must be strictly between 0 and 1")

    dataset_path = args.dataset
    if dataset_path.suffix == ".npz":
        metadata_path = dataset_path.with_suffix(".json")
    else:
        metadata_path = dataset_path / "human_bc_reconstructed.json"
        dataset_path = dataset_path / "human_bc_reconstructed.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(dataset_path) as data:
        obs = data["observations"]
        masks = data["masks"]
        actions = data["actions"].astype(np.int64)
        replay_ids = data["replay_ids"].astype(np.int32)
        seats = data["seats"].astype(np.int32)
        flags = data["flags"].astype(np.int32)
        ticks = data["ticks"].astype(np.int32)

    valid = (flags == 0) & (masks[np.arange(len(actions)), actions] == 1)
    rows = np.flatnonzero(valid)
    plays = rows[actions[rows] != 0]
    waits = rows[actions[rows] == 0]
    if not len(plays) or not len(waits):
        raise ValueError("BC dataset needs both legal play and WAIT labels")

    # Hold out entire replays, never individual rows from a match.
    replay_values = np.unique(replay_ids[rows])
    if len(replay_values) >= 2:
        validation_replay = replay_values[-1]
        train_rows = rows[replay_ids[rows] != validation_replay]
        val_rows = rows[replay_ids[rows] == validation_replay]
    else:
        train_rows, val_rows = rows, np.array([], dtype=np.int64)

    train_plays = train_rows[actions[train_rows] != 0]
    train_waits = train_rows[actions[train_rows] == 0]
    groups: dict[tuple[int, int], list[int]] = {}
    for index in rows:
        groups.setdefault((int(replay_ids[index]), int(seats[index])), []).append(int(index))
    for key in groups:
        groups[key].sort(key=lambda i: int(ticks[i]))
    positions = {
        key: {row: pos for pos, row in enumerate(indices)} for key, indices in groups.items()
    }

    schema = make_schema(metadata)
    adapter = CrForgeObservationAdapter(schema, len(metadata["vocabulary"]["names"]))
    cfg = ModelConfig.preset(args.model)
    cfg = ModelConfig(
        **{
            **cfg.__dict__,
            "context_len": args.context,
            "vocab_size": len(metadata["vocabulary"]["names"]),
        }
    )
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ActorCritic(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    def batch(rows_to_build: np.ndarray):
        windows = np.zeros((len(rows_to_build), args.context, schema.seat_ints), dtype=np.int32)
        time_mask = np.zeros((len(rows_to_build), args.context), dtype=np.bool_)
        legal = masks[rows_to_build].astype(bool)
        labels = actions[rows_to_build]
        for j, row in enumerate(rows_to_build):
            key = (int(replay_ids[row]), int(seats[row]))
            indices = groups[key]
            pos = positions[key][int(row)]
            seq = indices[max(0, pos - args.context + 1) : pos + 1]
            pad = args.context - len(seq)
            windows[j, pad:] = obs[seq]
            time_mask[j, pad:] = True
        header, entities, entity_mask = adapter(windows)
        hand_ids = obs[rows_to_build][:, list(schema.header["hand_card"])]
        return (
            torch.as_tensor(header, device=device),
            torch.as_tensor(entities, device=device),
            torch.as_tensor(entity_mask, device=device),
            torch.as_tensor(legal, device=device),
            torch.as_tensor(time_mask, device=device),
            torch.as_tensor(hand_ids, device=device),
            torch.as_tensor(labels, device=device),
        )

    def evaluate(rows_to_eval: np.ndarray) -> dict:
        if not len(rows_to_eval):
            return {}
        model.eval()
        losses, exact_correct, slot_correct, tile_correct = [], 0, 0, 0
        play_count, wait_count, wait_correct = 0, 0, 0
        with torch.no_grad():
            for start in range(0, len(rows_to_eval), args.batch_size):
                ids = rows_to_eval[start : start + args.batch_size]
                h, e, em, legal, tm, hand_ids, y = batch(ids)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device == "cuda" and torch.cuda.is_bf16_supported(),
                ):
                    logp, _ = model(h, e, em, legal, tm, hand_card_ids=hand_ids)
                    loss, _ = hierarchical_bc_loss(logp, y, legal, position_coef=args.position_coef)
                pred = logp.argmax(-1)
                losses.append(float(loss.cpu()))
                exact_correct += int((pred == y).sum().cpu())
                is_play = y != 0
                is_wait = ~is_play & (y < 2305)
                play_count += int(is_play.sum().cpu())
                wait_count += int(is_wait.sum().cpu())
                wait_correct += int(((pred == 0) & is_wait).sum().cpu())
                card_parts = [logp[..., :1]]
                for slot in range(4):
                    start = 1 + slot * 576
                    card_parts.append(
                        torch.logsumexp(logp[..., start : start + 576], dim=-1, keepdim=True)
                    )
                card_parts.append(logp[..., 2305:2307])
                card_logp = torch.cat(card_parts, dim=-1)
                target_card = torch.where(is_play, 1 + (y - 1) // 576, y.clamp(0, 0))
                target_card = torch.where(y >= 2305, 5 + y - 2305, target_card)
                pred_card = card_logp.argmax(-1)
                slot_correct += int(((pred_card == target_card) & is_play).sum().cpu())
                if is_play.any():
                    true_slot = ((y[is_play] - 1) // 576).long()
                    true_tile = ((y[is_play] - 1) % 576).long()
                    conditional = []
                    for slot in range(4):
                        start = 1 + slot * 576
                        conditional.append(
                            logp[..., start : start + 576] - card_logp[..., slot + 1 : slot + 2]
                        )
                    tile_logp = torch.stack(conditional, dim=-2)
                    row = torch.arange(len(y), device=device)[is_play]
                    tile_pred = tile_logp[row, true_slot].argmax(-1)
                    tile_correct += int((tile_pred == true_tile).sum().cpu())
        model.train()
        wait_accuracy = wait_correct / max(1, wait_count)
        play_slot_accuracy = slot_correct / max(1, play_count)
        return {
            "loss": float(np.mean(losses)),
            "exact_accuracy": exact_correct / len(rows_to_eval),
            "wait_accuracy": wait_accuracy,
            "balanced_wait_play_accuracy": 0.5 * (wait_accuracy + play_slot_accuracy),
            "play_slot_accuracy": play_slot_accuracy,
            "play_tile_accuracy": tile_correct / max(1, play_count),
            "validation_play_fraction": play_count / len(rows_to_eval),
            "rows": len(rows_to_eval),
        }

    started = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        n = min(args.samples_per_epoch, len(train_rows))
        n_play = max(1, int(n * args.play_fraction))
        n_wait = max(1, n - n_play)
        selected = np.concatenate(
            [
                rng.choice(train_plays, n_play, replace=n_play > len(train_plays)),
                rng.choice(train_waits, n_wait, replace=n_wait > len(train_waits)),
            ]
        )
        rng.shuffle(selected)
        model.train()
        epoch_losses, card_losses, position_losses = [], [], []
        for start in range(0, len(selected), args.batch_size):
            ids = selected[start : start + args.batch_size]
            h, e, em, legal, tm, hand_ids, y = batch(ids)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                logp, _ = model(h, e, em, legal, tm, hand_card_ids=hand_ids)
                loss, loss_parts = hierarchical_bc_loss(
                    logp, y, legal, position_coef=args.position_coef
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            card_losses.append(float(loss_parts["card_loss"].cpu()))
            position_losses.append(float(loss_parts["position_loss"].cpu()))
        report = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "card_loss": float(np.mean(card_losses)),
            "position_loss": float(np.mean(position_losses)),
            "validation": evaluate(val_rows),
        }
        history.append(report)
        print(json.dumps(report), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.__dict__,
            "dataset": str(dataset_path),
            "dataset_metadata": metadata,
            "history": history,
            "device": device,
        },
        args.out,
    )
    summary = {
        "checkpoint": str(args.out),
        "device": device,
        "parameters": model.parameter_count(),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "play_rows": len(train_plays),
        "wait_rows": len(train_waits),
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
