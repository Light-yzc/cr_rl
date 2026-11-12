import json

from cr_rl.monitoring.tensorboard_bridge import event_scalars, read_lines, summary_scalars


def test_json_tailer_does_not_lose_partial_append_and_handles_replace(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"update":1}\n{"update":')
    rows, cursor = read_lines(path, {})
    assert rows == [{"update": 1}]
    with path.open("ab") as stream:
        stream.write(b"2}\n")
    rows, cursor = read_lines(path, cursor)
    assert rows == [{"update": 2}]
    assert read_lines(path, cursor)[0] == []
    replacement = tmp_path / "replacement"
    replacement.write_text(json.dumps({"update": 3}) + "\n")
    replacement.replace(path)
    assert read_lines(path, cursor)[0] == [{"update": 3}]


def test_summary_uses_new_sample_throughput_and_keeps_precision_diagnostics():
    row = {
        "seconds": 10,
        "valid_learner_samples": 64,
        "completed_training_games": 2,
        "initial_logp_max_error": 0.0017,
        "peak_cuda_per_device_gib": {"cuda:1": 12},
        "metrics": {"value_loss": 0.01, "policy_loss": None, "card_kl": 0.004},
        "sample_utilization": {"critic_samples_committed": 128},
    }
    values = summary_scalars(row)
    assert values["performance/effective_new_samples_per_second"] == 6.4
    assert values["sample_utilization/critic_samples_committed"] == 128
    assert values["training/initial_logp_max_error"] == 0.0017
    assert values["ppo_update/value_loss"] == 0.01
    assert "ppo_update/policy_loss" not in values
    assert values["peak_gpu_memory_gib/cuda:1"] == 12


def test_actor_and_critic_curves_use_restored_optimizer_counters():
    step, values = event_scalars(
        {
            "phase": "optimization",
            "update": 145,
            "optimizer_steps": 2,
            "optimizer_steps_total": 2544,
            "loss": 0.2,
        }
    )
    assert step == 2544 and values["actor_step/loss"] == 0.2
    step, values = event_scalars(
        {
            "phase": "critic_optimization",
            "update": 145,
            "critic_optimizer_steps": 21000,
            "value_loss": 0.01,
        }
    )
    assert step == 21000 and values["critic_step/value_loss"] == 0.01


def test_summary_exports_adaptive_lr_evidence_and_skips_text_fields():
    row = {
        "seconds": 10,
        "valid_learner_samples": 64,
        "completed_training_games": 2,
        "actor_lr_schedule": {
            "enabled": True,
            "scope": "next_update",
            "lr_used": 1e-5,
            "next_lr": 5e-6,
            "reason": "kl_rejected",
            "groups_observed": 3,
            "samples_observed": 192,
            "rejected_groups": 1,
            "max_normalized_kl": 1.2,
            "low_kl_updates": 0,
            "completed_updates": 2,
        },
    }
    values = summary_scalars(row)
    assert values["actor_lr_schedule/lr_used"] == 1e-5
    assert values["actor_lr_schedule/next_lr"] == 5e-6
    assert values["actor_lr_schedule/max_normalized_kl"] == 1.2
    assert values["actor_lr_schedule/rejected_groups"] == 1
    assert values["actor_lr_schedule/samples_observed"] == 192
    assert values["actor_lr_schedule/completed_updates"] == 2
    assert "actor_lr_schedule/reason" not in values
    assert "actor_lr_schedule/scope" not in values


def test_ram_metrics_include_host_headroom_and_enforced_training_limits(tmp_path):
    from cr_rl.monitoring.tensorboard_bridge import host_memory_stats

    run = tmp_path / "run"
    proc = tmp_path / "proc"
    group = tmp_path / "cgroup" / "app" / "trainer"
    for path in (run, proc / "123", group):
        path.mkdir(parents=True)
    (run / "trainer.pid").write_text("123")
    (proc / "meminfo").write_text("MemAvailable: 8388608 kB\n")
    (proc / "123" / "status").write_text("VmRSS: 2097152 kB\nVmSwap: 262144 kB\n")
    (proc / "123" / "cgroup").write_text("0::/app/trainer\n")
    for field, value in {
        "memory.current": 3,
        "memory.peak": 5,
        "memory.max": 6,
        "memory.swap.current": 0,
        "memory.swap.max": 1,
    }.items():
        (group / field).write_text(str(value * 2**30))
    (group / "memory.events").write_text("oom_kill 0\n")
    metrics = host_memory_stats(run, proc, tmp_path / "cgroup")
    assert metrics["ram/host_available_gib"] == 8
    assert metrics["ram/trainer_rss_gib"] == 2
    assert metrics["ram/trainer_swap_gib"] == 0.25
    assert metrics["ram/service_limit_gib"] == 6
    assert metrics["ram/service_swap_limit_gib"] == 1
    assert metrics["ram/service_oom_kills"] == 0
    (run / "trainer.pid").unlink()
    assert host_memory_stats(run, proc, tmp_path / "cgroup") == {"ram/host_available_gib": 8}
