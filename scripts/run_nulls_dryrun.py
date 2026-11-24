"""Strictly read-only Nulls Royale -> Schema v4 -> architecture-v3 dry-run harness.

Connects to a live Nulls Royale (`16.635.1`) process via bounded `/proc/<pid>/mem`
reads, adapts snapshots to Schema v4 + legal masks, steps `StateInference` on
the recurrent `actor_export.pt` policy, and computes dry-run `TouchCalibration`
tap coordinates via `AdbActionBridge(..., execute=False)` without ever sending
live battle touch events.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cr_rl.client.adb_actions import (
    ActionObservation,
    AdbActionBridge,
    mumu_900x1600_calibration,
)
from cr_rl.client.nulls_reader import (
    DEFAULT_MAX_POLL_BYTES,
    DEFAULT_PACKAGE,
    NotInBattleError,
    NullsMemoryReader,
)
from cr_rl.client.schema_adapter import NullsToSchemaV4Adapter
from cr_rl.client.schema_v4 import SchemaV4
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.models.state_policy import state_model_from_checkpoint
from cr_rl.training.inference import StateInference


DEFAULT_SERIAL = "127.0.0.1:16416"
DEFAULT_CHECKPOINT = "artifacts/ppo_v3_v100/actor_export.pt"
DEFAULT_TABLES = r"C:\Users\Administrator\.crforge\tables\16.635.3"


def schema_from_model_config(cfg) -> SimpleNamespace:
    """Build a CrForge schema duck-type matching checkpoint ModelConfig fields."""
    header_fields = tuple(cfg.header_fields)
    entity_fields = tuple(cfg.entity_fields)
    header: dict[str, list[int]] = {}
    for index, name in enumerate(header_fields):
        if name != "reserved":
            header.setdefault(name, []).append(index)
    return SimpleNamespace(
        seat_ints=len(header_fields) + int(cfg.max_entities) * len(entity_fields),
        header_ints=len(header_fields),
        entity_ints=len(entity_fields),
        max_entities=int(cfg.max_entities),
        header={key: tuple(indices) for key, indices in header.items()},
        header_fields=header_fields,
        entity_fields=entity_fields,
    )


def format_action(action: int, hand_cards: tuple[str | None, ...]) -> str:
    """Format decoded Schema v4 action with hand card name when applicable."""
    decoded = SchemaV4.decode_action(action)
    kind = decoded["kind"]
    if kind == "wait":
        return "WAIT"
    if kind == "ability":
        return f"ABILITY(slot={decoded['slot']})"
    slot = int(decoded["slot"])
    card = hand_cards[slot] if 0 <= slot < len(hand_cards) else None
    return f"PLAY(slot={slot}, card={card or '?'}, col={decoded['col']}, row={decoded['row']})"


def load_policy_inference(
    checkpoint_path: str | Path,
    device: str,
    *,
    compute_value: bool = True,
    precision: str | None = None,
) -> tuple[torch.nn.Module, StateInference, int]:
    """Load an architecture-v3 actor checkpoint and construct StateInference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    return build_policy_inference(checkpoint, device, compute_value=compute_value, precision=precision)


def build_policy_inference(checkpoint, device, *, compute_value=True, precision=None):
    """Build inference from an already opened actor checkpoint."""
    model = state_model_from_checkpoint(checkpoint, device=device)
    model.deployment_metadata = {key: checkpoint.get(key) for key in
        ('updates','model_architecture_version','decision_ticks','backend_commit','data_version','content_sha')}
    model.eval()
    obs_schema = schema_from_model_config(model.cfg)
    if obs_schema.seat_ints != SchemaV4.SEAT_INTS:
        raise ValueError(
            f"Checkpoint schema seat_ints={obs_schema.seat_ints} differs from SchemaV4 ({SchemaV4.SEAT_INTS})"
        )
    obs_adapter = CrForgeObservationAdapter(
        obs_schema,
        vocabulary_size=int(model.cfg.vocab_size),
        entity_bucket=16,
    )
    infer = StateInference(
        model,
        obs_adapter,
        device=device,
        streams=1,
        compute_value=compute_value,
        precision=precision or ('fp16' if str(device).startswith('cuda') else 'fp32'),
    )
    return model, infer, int(getattr(model, "decision_ticks", 8))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Nulls Royale -> Schema v4 -> StateInference dry-run."
    )
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="ADB device serial.")
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="Android package name.")
    parser.add_argument("--adb", default="adb", help="Path to adb executable.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Actor checkpoint (.pt).")
    parser.add_argument("--tables", default=DEFAULT_TABLES, help="CrForge extracted CSV JSON tables dir.")
    parser.add_argument("--ticks", type=int, default=15, help="Maximum decision frames to sample.")
    parser.add_argument("--interval-ms", type=int, default=400, help="Polling interval in milliseconds.")
    parser.add_argument("--device", default="cpu", help="Torch device (e.g. cpu, cuda).")
    parser.add_argument(
        "--max-poll-bytes",
        type=int,
        default=DEFAULT_MAX_POLL_BYTES,
        help="Per-poll read budget in bytes (hard cap 4 MiB).",
    )
    parser.add_argument(
        "--own-seat",
        type=int,
        choices=(0, 1),
        default=None,
        help="Optional explicit seat override if local account ID does not match.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ticks < 1:
        raise ValueError("--ticks must be >= 1")
    if args.interval_ms < 50:
        raise ValueError("--interval-ms must be >= 50")

    state_adapter = NullsToSchemaV4Adapter(args.tables)
    model, infer, decision_ticks = load_policy_inference(args.checkpoint, args.device)
    vocab_size = len(state_adapter.vocab.names)
    if vocab_size != int(model.cfg.vocab_size):
        print(
            f"[dryrun] warning: tables vocab_size={vocab_size} != checkpoint vocab_size={model.cfg.vocab_size}",
            file=sys.stderr,
        )

    reader = NullsMemoryReader.connect(
        args.serial,
        package=args.package,
        adb=args.adb,
        max_poll_bytes=args.max_poll_bytes,
        require_running_clock=True,
    )
    calibration = mumu_900x1600_calibration(serial=args.serial, validated=False)
    latest_action_obs: ActionObservation | None = None
    bridge = AdbActionBridge(
        calibration,
        adb=args.adb,
        state_source=lambda: latest_action_obs,  # type: ignore[arg-type,return-value]
    )

    print(
        f"[dryrun] connected serial={args.serial} pid={reader.mem.pid} base=0x{reader.base:x} "
        f"device={args.device} decision_ticks={decision_ticks} execute=False (DRY-RUN)"
    )

    sampled = 0
    prev_tick = -1
    interval_s = args.interval_ms / 1000.0

    while sampled < args.ticks:
        step_t0 = time.monotonic()
        try:
            snap = reader.read_snapshot()
        except NotInBattleError as exc:
            if sampled == 0:
                print(f"[dryrun] NOT_IN_BATTLE: {exc}")
                return 2
            print(f"[dryrun] match_ended after {sampled} ticks: {exc}")
            return 0

        if prev_tick >= 0 and snap.tick < prev_tick:
            print(f"[dryrun] tick rewound ({prev_tick} -> {snap.tick}); resetting recurrent memory")
            infer.reset()
            state_adapter.reset_match()
        elif prev_tick >= 0 and snap.tick == prev_tick:
            time.sleep(min(interval_s, 0.2))
            continue
        prev_tick = snap.tick

        adapted = state_adapter.adapt(snap, own_seat=args.own_seat)
        match_id = f"nulls-{reader.mem.pid}-s{adapted.own_seat}"
        latest_action_obs = ActionObservation(
            match_id=match_id,
            observed_monotonic=time.monotonic(),
            hand_card_ids=adapted.hand_card_ids,
            legal_actions=tuple(bool(x) for x in adapted.legal_mask),
            width=calibration.width,
            height=calibration.height,
            verified=True,
        )

        raw_batch = adapted.obs[None, :]
        legal_batch = adapted.legal_mask[None, :]
        actions, _, values, stats = infer.sample(raw_batch, legal_batch)
        action = int(actions[0])

        expected_card_id = None
        if 1 <= action <= 2304:
            slot = (action - 1) // 576
            expected_card_id = int(adapted.hand_card_ids[slot])

        # Strictly dry-run: execute=False never invokes ADB taps.
        receipt = bridge.submit(
            action,
            match_id=match_id,
            expected_card_id=expected_card_id,
            execute=False,
        )

        kib = reader.mem.poll_bytes_read / 1024.0
        elixir = float(adapted.obs[SchemaV4.H_ELIXIR]) / SchemaV4.ELIXIR_SCALE
        crowns_own = int(adapted.obs[SchemaV4.H_CROWNS])
        crowns_enemy = int(adapted.obs[SchemaV4.H_CROWNS_ENEMY])
        entity_count = int(adapted.obs[SchemaV4.H_ENTITY_COUNT])
        legal_count = int(adapted.legal_mask.sum())
        value_str = f"{float(values[0]):+.3f}" if values is not None else "n/a"
        entropy = stats.get("entropy")
        entropy_str = f"{float(entropy):.3f}" if entropy is not None else "n/a"
        action_str = format_action(action, adapted.hand_cards)
        unmatched = ",".join(adapted.unmatched_names) if adapted.unmatched_names else "-"

        print(
            f"[tick={adapted.tick:4d} {adapted.time_ms:6d}ms] "
            f"seat={adapted.own_seat} KiB={kib:5.1f} elixir={elixir:4.2f} "
            f"crowns={crowns_own}-{crowns_enemy} ents={entity_count:2d} legal={legal_count:4d} "
            f"val={value_str} H={entropy_str} act={action_str} taps={receipt['taps']} "
            f"unmatched={unmatched}"
        )

        sampled += 1
        elapsed = time.monotonic() - step_t0
        if sampled < args.ticks and elapsed < interval_s:
            time.sleep(interval_s - elapsed)

    print(
        f"[dryrun] completed ticks={sampled} unmatched={sorted(state_adapter.unmatched_names)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
