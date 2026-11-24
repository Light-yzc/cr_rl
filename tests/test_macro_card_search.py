"""Unit and native CrForge integration tests for universal 4x4 MacroCardSearch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

if sys.platform == "win32" and not os.environ.get("JAVA_HOME") and Path(r"F:\JDK\bin\java.exe").exists():
    os.environ["JAVA_HOME"] = r"F:\JDK"

import numpy as np
import torch

from crforge_env import _jvm
from cr_rl.config import SimulatorPaths
from cr_rl.data.observations import CrForgeObservationAdapter
from cr_rl.envs.crforge import CrForgeVector
from cr_rl.models import ModelConfig
from cr_rl.models.state_policy import StateActorCritic
from cr_rl.models.state_temporal import StateTemporalConfig
from cr_rl.training.colab_pipeline import ppo_command, progress
from cr_rl.training.macro_card_search import MacroCardSearch, combo_frame_weights, probe_anchor_weights
from cr_rl.training.state_selfplay import _extract_decks


def test_combo_frame_weights_and_pipeline_flags(tmp_path: Path):
    actions = np.array([0, 0, 100, 0, 200, 0], dtype=np.int64)
    keep = np.array([False, True, True, True, True, True], dtype=bool)
    mplay = np.array([False, False, True, False, False, False], dtype=bool)
    mwait = np.array([False, True, False, False, False, False], dtype=bool)
    w = combo_frame_weights(actions, keep, mplay, mwait)
    np.testing.assert_allclose(w, np.array([0.0, 0.5, 1.5, 0.06, 1.0, 0.06], dtype=np.float32))

    pw = probe_anchor_weights({"actions": actions, "actor_keep": keep})
    np.testing.assert_allclose(pw, np.array([0.0, 0.06, 1.0, 0.06, 1.0, 0.06], dtype=np.float32))

    # Verify colab_pipeline ppo_command & progress integration
    (tmp_path / "prepared/input").mkdir(parents=True)
    (tmp_path / "prepared/ppo_latest.pt").touch()
    (tmp_path / "prepared/input/ppo_latest.pt").touch()
    (tmp_path / "prepared/anchor_actor.pt").touch()
    (tmp_path / "prepared/source_league.yaml").touch()
    (tmp_path / "prepared/source_metadata.json").write_text(json.dumps({"training_args": {"gamma": 0.9995}}))
    cfg = {
        "run_name": "test_macro",
        "expert_bc": False,
        "awbc": False,
        "macro_search": True,
        "macro_search_interval": 5,
        "macro_search_offset": 2,
        "macro_search_candidates": 2,
        "macro_search_win_margin": 0.25,
        "macro_search_learning_rate": 1e-6,
        "macro_search_max_kl": 0.005,
    }
    cmd = ppo_command(tmp_path, cfg)
    assert "--macro-search" in cmd
    idx = cmd.index("--macro-search-interval")
    assert cmd[idx + 1] == "5"
    assert cmd[cmd.index("--macro-search-offset") + 1] == "2"
    idx_m = cmd.index("--macro-search-win-margin")
    assert cmd[idx_m + 1] == "0.25"

    run_dir = tmp_path / "runs/test_macro"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"update": 10, "macro_search": {"ran": True, "accepted": True}}))
    prog = progress(tmp_path, "test_macro")
    assert prog["macro_search"] == {"ran": True, "accepted": True}

    ms_strat = MacroCardSearch(tmp_path / "strat_check", seed=42)
    min_steps, split = ms_strat._sample_lane_min_steps(380, fast_cpu=False)
    assert split == 9
    assert np.all((min_steps[:9] >= 10) & (min_steps[:9] < 180))
    assert np.all((min_steps[9:] >= 180) & (min_steps[9:] <= 350))


def test_native_macro_card_search_end_to_end_and_distillation(tmp_path: Path):
    sim_paths = SimulatorPaths.discover()
    _jvm.start(
        sim_paths.install,
        jvm_args=("-Xmx256m", "-Xms64m", "-XX:MaxMetaspaceSize=96m", "-XX:CompressedClassSpaceSize=32m", "-XX:+UseSerialGC"),
    )
    with CrForgeVector(2, 2, paths=sim_paths, independent=True, arena="Arena_L", decision_ticks=8) as env:
        schema = env.vector.schema
        vocab_size = env.vector.vocabulary_size
        raw_sample = env.observations[:, 0].copy()
        mask_sample = env.masks[:, 0].copy()

    torch.manual_seed(7)
    cfg = ModelConfig(
        entity_dim=16,
        entity_heads=2,
        temporal_dim=16,
        temporal_heads=2,
        temporal_layers=2,
        vocab_size=vocab_size,
        header_fields=tuple(schema.header_fields),
        entity_fields=tuple(schema.entity_fields),
        representation_version=2,
        readout_version=2,
    )
    temporal = StateTemporalConfig(
        width=16,
        heads=2,
        kv_heads=2,
        head_dim=8,
        pattern=("kda", "gqa"),
        ffn_multiplier=2,
        chunk_size=4,
        cache_initial_capacity=4,
    )
    actor = StateActorCritic(cfg, temporal, frame_microbatch=16).eval()
    adapter = CrForgeObservationAdapter(schema, vocab_size, entity_bucket=16)

    # Create synthetic PPO probes from real native schema frames
    probe_obs = np.repeat(raw_sample[:1], 12, axis=0)
    probe_mask = np.repeat(mask_sample[:1], 12, axis=0)
    probe_mask[:, :10] = True
    probes = [
        dict(
            raw_obs=probe_obs,
            legal_masks=probe_mask,
            actions=np.array([0, 1, 0, 2, 0, 0, 3, 0, 0, 1, 0, 0], dtype=np.int64),
            actor_keep=np.ones(12, dtype=bool),
        )
        for _ in range(2)
    ]

    _, deck_names, decks, forms = _extract_decks(Path("configs/decks_all51.yaml"))
    sparky_idx = deck_names.index("user_giant_sparky_pump_plain")
    sub_names = [deck_names[sparky_idx], deck_names[0]]
    sub_decks = [decks[sparky_idx], decks[0]]
    sub_forms = [forms[sparky_idx], forms[0]]

    ms = MacroCardSearch(tmp_path, max_buffer_episodes=8, seed=123)
    # Use win_margin=0.0 so even a tied/slightly positive combo exercises admission + distillation + gates
    res = ms.run_phase(
        actor,
        adapter,
        probes,
        env_paths=sim_paths,
        decks=sub_decks,
        forms=sub_forms,
        deck_names=sub_names,
        update=5,
        device=torch.device("cpu"),
        precision="fp32",
        candidates=1,
        win_margin=0.0,
        batch_episodes=2,
        lr=1e-6,
        max_kl=0.005,
    )
    assert res["ran"] is True
    assert res["candidates_found"] == 1
    assert len(res["branch_reports"]) == 1

    # Also test state_dict persistence and forced buffer distillation + KL/WAIT gates
    while len(ms.entries) < 2:
        # Ensure >=2 combo episodes in buffer so forced non-admitting phase distills from buffer
        ms._save_episode(
            dict(
                raw_obs=probe_obs,
                legal_masks=probe_mask,
                actions=np.array([0, 2, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0], dtype=np.int64),
                actor_keep=np.ones(12, dtype=bool),
                weights=np.array([0.5, 1.5, 0.5, 1.5, 0.06, 0.06, 1.0, 0.06, 0.06, 1.0, 0.06, 0.06], dtype=np.float32),
            ),
            dict(update=5, deck=sub_names[0], combo_cards=["ZapMachine", "Giant"], win_advantage=0.5),
        )

    saved = ms.state_dict()
    restored = MacroCardSearch(tmp_path, max_buffer_episodes=8, state=saved)
    assert len(restored.entries) == len(ms.entries) >= 1

    res2 = restored.run_phase(
        actor,
        adapter,
        probes,
        env_paths=sim_paths,
        decks=sub_decks,
        forms=sub_forms,
        deck_names=sub_names,
        update=10,
        device=torch.device("cpu"),
        precision="fp32",
        candidates=1,
        win_margin=0.99,  # force distillation from existing buffer entries
        batch_episodes=2,
        lr=1e-6,
        max_kl=0.005,
    )
    assert res2["ran"] is True
    assert res2["actor_attempted"] is True
    assert res2["ppo_forward_kl"] <= 0.005
    assert abs(res2["wait_prob_delta"]) <= 0.005
