"""Fixed-budget real simulator/Actor/opponent/Critic throughput configuration grid."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import ActorPolicy, HistoryValueNetwork, ModelConfig
from cr_rl.models.incremental_frames import IncrementalFrameHistory
from cr_rl.training.inference import FrozenInference
from cr_rl.training.persistence import atomic_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--opponent", required=True, type=Path)
    parser.add_argument("--critic", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")

    def actor_from(path):
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        model = ActorPolicy(ModelConfig(**state["config"])).cuda().eval()
        model.load_state_dict(state["model"])
        return model

    actor, opponent = actor_from(args.actor), actor_from(args.opponent)
    state = torch.load(args.critic, map_location="cpu", mmap=True, weights_only=False)
    critic = HistoryValueNetwork(ModelConfig(**state["critic_config"])).cuda().eval()
    critic.load_state_dict(state["critic"])
    league = yaml.safe_load(Path("configs/decks_initial3.yaml").read_text())
    decks = list(league["decks"].values())
    results = []
    for repeat in range(3):
        for count in (2, 4, 8) if repeat % 2 == 0 else (8, 4, 2):
            steps = 512 // count
            torch.manual_seed(20261105 + repeat)
            with CrForgeVector(count, count, independent=True, arena="Arena_L") as env:
                env.set_autoreset(False)
                episode_ids, next_episode = np.arange(count), count
                for lane in range(count):
                    env.vector.reset_env(
                        lane,
                        20261105 + repeat * 10000 + lane,
                        decks[lane % 3],
                        decks[(lane + 1) % 3],
                        [0] * 8,
                        [0] * 8,
                    )
                adapter = CrForgeObservationAdapter(
                    env.vector.schema, env.vector.vocabulary_size, entity_bucket=16
                )
                history = IncrementalFrameHistory(
                    count * 2, actor.cfg.context_len, adapter.schema.seat_ints
                )
                actor_inf, opponent_inf, critic_inf = (
                    FrozenInference(m, adapter, "cuda") for m in (actor, opponent, critic)
                )
                # A fixed 50/50 current/anchor mix gives exactly 768 learner rows
                # for all three counts, matching v2's expected current fraction.
                is_current = np.repeat(np.arange(count) % 2 == 0, 2)
                designated = np.arange(count) * 2 + np.arange(count) % 2
                is_current[designated] = True
                current_rows, historical_rows = (
                    np.flatnonzero(is_current),
                    np.flatnonzero(~is_current),
                )
                learner_samples = games = 0
                entity_sum = history_sum = 0.0
                torch.cuda.synchronize()
                began = time.perf_counter()
                for _step in range(steps):
                    raw, masks = env.observations.copy(), env.masks.copy()
                    flat_masks = masks.reshape(count * 2, -1)
                    history.append(raw.reshape(count * 2, -1), np.repeat(episode_ids, 2))
                    request = history.request()
                    actions = np.zeros(count * 2, dtype=np.int32)
                    for inference, rows in (
                        (actor_inf, current_rows),
                        (opponent_inf, historical_rows),
                    ):
                        actions[rows] = inference.sample_incremental(
                            request.select(rows), flat_masks[rows]
                        )[0]
                    values = critic_inf.evaluate_incremental(request).cpu().numpy()
                    if (
                        not np.isfinite(values).all()
                        or not np.take_along_axis(flat_masks, actions[:, None], -1).all()
                    ):
                        raise RuntimeError("Invalid frozen rollout benchmark output")
                    entity_sum += float(raw[..., adapter.schema.header["entity_count"][0]].mean())
                    history_sum += float(
                        np.minimum(history.positions + 1, actor.cfg.context_len).mean()
                    )
                    learner_samples += len(current_rows)
                    env.step(actions.reshape(count, 2))
                    for lane in np.flatnonzero(env.dones):
                        info = env.vector.info(int(lane))
                        if info["fault"] or info["truncated"]:
                            raise RuntimeError("Simulator fault in rollout configuration benchmark")
                        games += 1
                        episode_ids[lane] = next_episode
                        next_episode += 1
                        env.vector.reset_env(
                            int(lane),
                            20271105 + next_episode,
                            decks[lane % 3],
                            decks[(lane + 1) % 3],
                            [0] * 8,
                            [0] * 8,
                        )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                assert learner_samples == 768 and steps * count == 512
                row = dict(
                    envs=count,
                    repeat=repeat,
                    steps_per_env=steps,
                    environment_transitions=512,
                    learner_samples=learner_samples,
                    seconds=elapsed,
                    learner_samples_per_second=learner_samples / elapsed,
                    terminal_games=games,
                    mean_entities=entity_sum / steps,
                    mean_history_length=history_sum / steps,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
    atomic_json(
        args.out,
        dict(
            results=results,
            medians={
                str(n): float(
                    np.median([r["learner_samples_per_second"] for r in results if r["envs"] == n])
                )
                for n in (2, 4, 8)
            },
            actor_sha256=file_sha256(args.actor),
            critic_sha256=file_sha256(args.critic),
            limitation=(
                "Frozen mixed-policy collector only; different histories/RNG layouts, "
                "50% current mix, no optimization or backups"
            ),
        ),
    )


if __name__ == "__main__":
    main()
