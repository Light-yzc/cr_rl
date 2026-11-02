import json
from types import SimpleNamespace

import numpy as np

from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.data.trajectory import TrajectoryBuffer


def test_trajectory_buffer_roundtrip(tmp_path):
    buffer = TrajectoryBuffer({"schema_version": 4, "seed": 7})
    obs = np.zeros((2, 2, 7024), dtype=np.int32)
    masks = np.zeros((2, 2, 2307), dtype=np.uint8)
    masks[..., 0] = 1
    actions = np.zeros((2, 2), dtype=np.int32)
    rewards = np.zeros((2, 2), dtype=np.float32)
    dones = np.zeros(2, dtype=np.uint8)
    buffer.append(obs, masks, actions, rewards, obs, masks, dones)
    path = buffer.save(tmp_path / "rollout.npz")
    with np.load(path) as data:
        assert data["obs"].shape == (1, 2, 2, 7024)
        assert data["next_masks"].shape == (1, 2, 2, 2307)
        assert data["old_log_probs"].shape == (1, 2, 2)
        assert data["learner_mask"].shape == (1, 2, 2)
        assert data["terminated"].shape == (1, 2)
        assert not bool(
            json.loads(path.with_suffix(".npz.json").read_text())["provenance_complete"]
        )
        assert not data["valid"].any()  # legacy collection is not silently PPO-eligible


def test_trajectory_v3_provenance_and_episode_local_invalidation(tmp_path):
    import json

    buffer = TrajectoryBuffer({"run_id": "test"})
    obs = np.zeros((2, 2, 7024), dtype=np.int32)
    masks = np.zeros((2, 2, 2307), dtype=np.uint8)
    masks[..., 0] = 1
    actions = np.zeros((2, 2), dtype=np.int32)
    rewards = np.zeros((2, 2), dtype=np.float32)
    dones = np.zeros(2, dtype=np.uint8)
    for _step in range(3):
        buffer.append(
            obs,
            masks,
            actions,
            rewards,
            obs,
            masks,
            dones,
            terminated=dones,
            truncated=dones,
            valid=np.ones((2, 2), dtype=np.uint8),
            action_consumed=np.ones((2, 2), dtype=np.uint8),
            action_refusals=np.zeros((2, 2), dtype=np.int32),
            run_refusals=np.zeros((2, 2), dtype=np.int32),
            old_log_probs=np.zeros((2, 2), dtype=np.float32),
            old_values=np.zeros((2, 2), dtype=np.float32),
            bootstrap_values=np.zeros((2, 2), dtype=np.float32),
            bootstrap_valid=np.ones((2, 2), dtype=np.uint8),
            learner_mask=np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
            policy_ids=np.zeros((2, 2), dtype=np.int32),
            opponent_ids=np.ones((2, 2), dtype=np.int32),
            policy_versions=np.zeros((2, 2), dtype=np.int32),
            episode_ids=np.asarray([0, 1], dtype=np.int64),
            episode_start=np.asarray([1, 1], dtype=np.uint8),
            episode_steps=np.asarray([0, 0], dtype=np.int32),
            ticks=np.asarray([0, 0], dtype=np.int32),
            winner=np.asarray([-1, -1], dtype=np.int8),
            end_reason=np.asarray([0, 0], dtype=np.uint8),
            fault_codes=np.zeros(2, dtype=np.int16),
            seeds=np.asarray([10, 20], dtype=np.int64),
        )
    buffer.invalidate_episode(env_index=0, start_step=1, end_step=2)
    path = buffer.save(tmp_path / "selfplay.npz")
    metadata = json.loads(path.with_suffix(".npz.json").read_text())
    with np.load(path) as data:
        assert metadata["trajectory_schema_version"] == 3
        assert metadata["provenance_complete"] is True
        assert data["valid"][:, 0].sum() == 2
        assert data["valid"][:, 1].sum() == 6
        assert data["bootstrap_valid"][:, 0].sum() == 2
        assert data["bootstrap_valid"][:, 1].sum() == 6


def test_observation_adapter_normalizes_units_and_keeps_empty_frame_safe():
    header_fields = ["entity_count", "elixir", "hand_card", "pending_x"]
    entity_fields = ["id", "kind", "side", "row", "x", "hp"]
    entity_fields += [f"unused_{i}" for i in range(28)]
    schema = SimpleNamespace(
        seat_ints=7024,
        header_ints=224,
        entity_ints=34,
        max_entities=200,
        header={"entity_count": (0,)},
        header_fields=header_fields,
        entity_fields=entity_fields,
    )
    raw = np.zeros((2, 7024), dtype=np.int32)
    raw[0, 0] = 1
    raw[0, 1] = 50000
    raw[0, 2] = 100
    raw[0, 3] = 9000
    raw[0, 224 + 3] = 100
    raw[0, 224 + 4] = 9000
    raw[0, 224 + 5] = 1000
    header, entities, mask = CrForgeObservationAdapter(schema, 1000)(raw)
    assert header[0, 1] == 0.5
    assert header[0, 2] == 0.1
    assert header[0, 3] == 0.5
    assert entities[0, 0, 3] == 0.1
    assert entities[0, 0, 4] == 0.5
    assert entities[0, 0, 5] == 0.1
    assert mask[0, 0]
    assert mask[1, 0]  # dummy key prevents all-masked attention on padding frames
