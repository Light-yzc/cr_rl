"""Unit and native CrForge integration tests for LookaheadDistiller."""

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
from cr_rl.models.state_policy import StateActorCritic, StateValueNetwork
from cr_rl.models.state_temporal import StateTemporalConfig
from cr_rl.models.learned_placement import PlacementAdapter, definition_for
from cr_rl.training.colab_pipeline import ppo_command, progress
from cr_rl.training.lookahead_distiller import LookaheadDistiller, lookahead_frame_weights, unfreeze_distill_heads


def test_lookahead_frame_weights_and_pipeline_flags(tmp_path: Path):
    actions = np.array([0, 120, 0, 650, 0, 12], dtype=np.int64)
    keep = np.array([False, True, True, True, True, True], dtype=bool)
    searched = np.array([False, False, True, True, True, True], dtype=bool)
    overridden = np.array([False, False, False, False, True, True], dtype=bool)

    w = lookahead_frame_weights(actions, keep, searched, overridden)
    np.testing.assert_allclose(w, np.array([0.0, 0.35, 0.20, 0.50, 1.80, 2.50], dtype=np.float32))

    bank_mask = np.array([False, False, True, False, True, False], dtype=bool)
    w_bank = lookahead_frame_weights(actions, keep, searched, overridden, override_boost=1.5, bank_wait_mask=bank_mask)
    np.testing.assert_allclose(w_bank, np.array([0.0, 0.35, 3.75, 0.50, 3.75, 3.75], dtype=np.float32))

    adv_mask = np.array([False, False, False, False, True, True], dtype=bool)
    w_adv_only = lookahead_frame_weights(
        actions, keep, searched, overridden, override_boost=1.5, adverse_mask=adv_mask, override_only=True
    )
    np.testing.assert_allclose(w_adv_only, np.array([0.0, 0.0, 0.0, 0.0, 2.70 * 1.6, 3.75 * 1.6], dtype=np.float32))

    # Verify colab_pipeline ppo_command & progress integration
    (tmp_path / "prepared/input").mkdir(parents=True)
    (tmp_path / "prepared/input/ppo_latest.pt").touch()
    (tmp_path / "prepared/anchor_actor.pt").touch()
    (tmp_path / "prepared/source_league.yaml").touch()
    (tmp_path / "prepared/source_metadata.json").write_text(json.dumps({"training_args": {"gamma": 0.9995}}))
    (tmp_path / "prepared/full_dataset.json").write_text(json.dumps({"local_path": str(tmp_path / "ds")}))

    cmd = ppo_command(
        tmp_path,
        {
            "run_name": "r1",
            "lookahead_distill": True,
            "lookahead_distill_interval": 3,
            "lookahead_distill_offset": 1,
            "lookahead_distill_games": 3,
            "lookahead_distill_horizon": 10,
            "lookahead_distill_margin": 0.008,
        },
    )
    assert "--lookahead-distill" in cmd
    assert "--lookahead-distill-interval" in cmd

    run_dir = tmp_path / "runs/r1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"update": 1522, "lookahead_distill": {"ran": True, "accepted": True}})
    )
    prog = progress(tmp_path, "r1")
    assert prog["lookahead_distill"] == {"ran": True, "accepted": True}


def test_lookahead_distiller_end_to_end_native_and_distillation(tmp_path: Path):
    sim_paths = SimulatorPaths.discover()
    _jvm.start(
        sim_paths.install,
        jvm_args=("-Xmx512m", "-Xms128m", "-XX:+UseG1GC"),
    )
    with CrForgeVector(2, 2, paths=sim_paths, independent=True, arena="Arena_L", decision_ticks=8) as env:
        schema = env.vector.schema
        vocab_size = env.vector.vocabulary_size
        raw_sample = env.observations[:, 0].copy()
        mask_sample = env.masks[:, 0].copy()

    torch.manual_seed(11)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = StateActorCritic(cfg, temporal, frame_microbatch=16).to(device).eval()
    actor.ppo_actor_fp32 = True  # Regression guard for cr-l4 mixed fp32-actor + fp16-critic!
    critic = StateValueNetwork(cfg, temporal, decision_ticks=8, frame_microbatch=16).to(device).eval()
    adapter = CrForgeObservationAdapter(schema, vocab_size, entity_bucket=16)

    probe_obs = np.repeat(raw_sample[:1], 12, axis=0)
    probe_mask = np.repeat(mask_sample[:1], 12, axis=0)
    probe_mask[:, :10] = True
    probe_ep = dict(
        raw_obs=probe_obs,
        legal_masks=probe_mask,
        actions=np.array([0, 1, 0, 2, 0, 0, 3, 0, 0, 1, 0, 0], dtype=np.int64),
        actor_keep=np.ones(12, dtype=bool),
    )
    probes = [probe_ep, probe_ep]

    decks = [
        ["ZapMachine", "Elixir Collector", "Wizard", "Rage", "Giant", "DarkPrince", "MiniPekka", "MinionHorde"],
        ["Cannon", "MightyMiner", "Firecracker", "Log", "HogRider", "ElectroSpirit", "Earthquake", "Skeletons"],
    ]
    forms = [[0] * 8, [1, 0, 1, 0, 0, 0, 0, 0]]
    deck_names = ["user_giant_sparky_pump_horde_plain", "human_popular_02"]

    distiller = LookaheadDistiller(tmp_path / "ld_buf", max_buffer_episodes=8, seed=123)
    synth_weights = lookahead_frame_weights(
        probe_ep["actions"],
        probe_ep["actor_keep"],
        np.ones(12, dtype=bool),
        np.ones(12, dtype=bool),
    )
    distiller._save_episode(
        dict(**probe_ep, weights=synth_weights),
        dict(update=0, deck=deck_names[0], opp_deck=deck_names[1], crown_diff=2, overrides=3, override_cards=["ZapMachine"]),
    )
    distiller._save_episode(
        dict(**probe_ep, weights=synth_weights),
        dict(update=0, deck=deck_names[0], opp_deck=deck_names[1], crown_diff=1, overrides=2, override_cards=["Giant"]),
    )

    res = distiller.run_phase(
        actor,
        critic,
        adapter,
        probes,
        env_paths=sim_paths,
        decks=decks,
        forms=forms,
        deck_names=deck_names,
        update=1,
        device=device,
        precision="fp16" if device.type == "cuda" else "fp32",
        games=4,
        parallel_games=3,
        max_parallel_searches=2,
        candidates=6,
        rollouts=2,
        horizon_steps=3,
        trigger_elixir=5.0,
        override_margin=0.0,
        max_searches_per_game=2,
        max_game_steps=36,
        batch_episodes=2,
        optimizer_steps=2,
        lr=2.5e-6,
        max_kl=0.005,
        focal_deck_indices=[0],
    )
    assert res["ran"] is True
    assert res["actor_attempted"] is True
    assert res["optimizer_steps"] >= 1
    assert res["ppo_forward_kl"] is not None and res["ppo_forward_kl"] <= 0.005
    assert len(res["game_reports"]) == 4
    assert res["parallel_games"] == 3
    assert res["search_lanes"] == 24

    restored = LookaheadDistiller(tmp_path / "ld_buf", max_buffer_episodes=8, state=distiller.state_dict())
    assert restored.last_update == 1
    assert len(restored.entries) == len(distiller.entries)
    assert any(k.startswith("tile_actor.") for k in res["changed_parameters"]) or not res["accepted"]

    # Also verify PlacementAdapter (`card_prior_cnn_001` with frozen `.anchor`, matching L4 production actor!)
    lp_def = definition_for(actor, "card_prior_cnn_001", cnn_channels=32)
    actor.add_module("learned_placement", PlacementAdapter(actor, lp_def))
    actor.learned_placement_definition = dict(lp_def)
    unfrozen_heads_only = unfreeze_distill_heads(actor, include_placement=True, include_last_temporal=False)
    assert any(n.startswith("slot_actor.") for n in unfrozen_heads_only)
    assert any(n.startswith("learned_placement.") for n in unfrozen_heads_only)
    assert not any(n.endswith(".anchor") for n in unfrozen_heads_only)
    assert not any(n.startswith("temporal.") for n in unfrozen_heads_only)

    unfrozen_full = unfreeze_distill_heads(actor, include_placement=True, include_last_temporal=True)
    assert any(n.startswith("temporal.blocks.1.") for n in unfrozen_full)
    assert any(n.startswith("temporal.final_norm.") for n in unfrozen_full)
    assert not any(n.startswith("temporal.blocks.0.") for n in unfrozen_full)
