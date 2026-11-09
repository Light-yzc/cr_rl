import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import benchmark_replay_repairs as benchmark
from scripts.benchmark_replay_repairs import fixed_cohort, stratified_summary, summarize


def test_fixed_cohort_preserves_declared_rows_and_tags(tmp_path):
    source = tmp_path / "source.parquet"
    rows = [
        dict(replay_tag=str(i), battle_type="ranked", game_mode="ranked", payload_json="{}")
        for i in range(10)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    target = tmp_path / "cohort.parquet"
    result = fixed_cohort(source, target, start=2, count=3)
    assert result["replay_tags"] == ["2", "3", "4"]
    assert pq.read_table(target).to_pylist() == rows[2:5]


def test_summary_never_equates_event_completion_with_terminal_or_missing_rows():
    records = [
        {
            "accepted": True, "used": 20, "total_play_events": 20,
            "source_event_stream_completed": True, "all_source_events_executed": True,
            "simulator_terminated": False,
        },
        {
            "accepted": False, "used": 8, "total_play_events": 20,
            "source_event_stream_completed": False, "all_source_events_executed": False,
            "simulator_terminated": True,
        },
    ]
    result = summarize(records, 2)
    assert result["source_event_stream_completed"] == 1
    assert result["simulator_terminals"] == 1
    assert result["all_source_events_execution_rate"] == 0.5
    assert result["native_successful_commands"] is None
    partial = summarize(records, 3)
    assert partial["source_event_stream_completion_rate"] == 1 / 3
    assert partial["all_source_events_execution_rate_attempted"] == 0.5
    assert partial["rows_without_attempt_record"] == 1
    # Legacy metadata's absent cutoff or coverage=1 is no proof of completion.
    legacy = summarize([dict(used=20, total_play_events=20, coverage=1)], 1)
    assert legacy["source_event_stream_completed"] is None
    assert legacy["simulator_terminals"] is None
    json.dumps(legacy, allow_nan=False)


def test_stratified_diagnostics_keep_raw_source_denominator_with_compatibility_rejections():
    traits = [
        dict(source_row_index=0, source_events=40, last_source_tick=3000,
             has_hero_marker=False, has_evolution_marker=False),
        dict(source_row_index=1, source_events=50, last_source_tick=4000,
             has_hero_marker=True, has_evolution_marker=True),
        dict(source_row_index=2, source_events=8, last_source_tick=900,
             has_hero_marker=False, has_evolution_marker=True),
    ]
    records = [dict(source_row_index=0, all_source_events_executed=True)]
    result = stratified_summary(records, traits)
    assert result["long_source"]["source_rows"] == 2
    assert result["long_source"]["complete_per_source_row"] == 0.5
    assert result["long_source"]["complete_per_attempt"] == 1
    assert result["has_hero_marker"]["complete"] == 0
    assert result["plain_markers"]["complete"] == 1


def test_early_terminal_candidate_requires_native_normal_end_and_counts_pending_separately():
    early = dict(early_terminal_before_last_source_event=True, simulator_match_decided=True,
                 simulator_fault=False, simulator_truncated=False,
                 stop_reason="native_match_decided", pending_commands_settled=False)
    rows = [early, {**early, "simulator_fault": True},
            {**early, "simulator_truncated": True},
            {**early, "stop_reason": "native_run_refusal"},
            {"all_source_events_executed": True}]
    result = summarize(rows, 5)
    assert result["normal_early_terminal_episodes"] == 1
    assert result["normal_early_terminal_episodes_with_pending_to_mask"] == 1
    assert result["full_source_or_normal_early_terminal_candidates"] == 2


def test_explicit_engine_terminal_and_winner_override_legacy_inference():
    base = dict(early_terminal_before_last_source_event=True, simulator_match_decided=True,
                simulator_fault=False, simulator_truncated=False,
                stop_reason="native_match_decided", pending_commands_settled=True,
                normal_simulator_terminal=True, native_winner=1,
                source_or_native_terminal_completed=True)
    result = summarize([base, {**base, "native_winner": -1,
                              "normal_simulator_terminal": False,
                              "source_or_native_terminal_completed": False}], 2)
    assert result["normal_early_terminal_episodes"] == 1
    assert result["normal_simulator_terminal"] == 1
    assert result["source_or_native_terminal_completed"] == 1
    assert result["source_or_native_terminal_completion_rate_raw"] == 0.5


def test_winner_histogram_excludes_native_default_for_undecided_matches():
    rows = [dict(native_winner=0, normal_simulator_terminal=False),
            dict(native_winner=0, normal_simulator_terminal=True),
            dict(native_winner=1, normal_simulator_terminal=True)]
    result = summarize(rows, 3)
    assert result["native_winner_counts"] == {"0": 1, "1": 1}
    assert result["raw_native_winner_values"] == {"0": 2, "1": 1}


def test_code_fingerprints_include_patch_tree_and_missing_declared_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    patch = tmp_path / "patches/crforge/late_hero.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("before")
    initial = benchmark.code_fingerprints()
    assert initial["src/cr_rl/envs/backend_patch.py"] is None
    assert initial["scripts/manage_crforge_patch.py"] is None
    assert initial["configs/backends.yaml"] is None
    patch.write_text("after")
    updated = benchmark.code_fingerprints()
    assert initial["patches/crforge/late_hero.patch"] != updated["patches/crforge/late_hero.patch"]


def test_backend_fingerprints_record_real_binaries_and_verify_receipt_without_jvm(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    vendor = tmp_path / ".cache/vendor/crforge"
    install = vendor / "env/build/install/env"
    library = install / "lib"
    library.mkdir(parents=True)
    for name in ("core.jar", "env.jar", "air.jar", "annotations.jar", "json-core.jar", "json.jar"):
        (library / name).write_text(name)
    git_values = {
        ("rev-parse", "HEAD"): b"7ee59b39\n",
        ("status", "--porcelain"): b" M source.java\n",
        ("diff", "--binary", "HEAD"): b"actual patch bytes",
        ("ls-files", "--others", "--exclude-standard", "-z"): b"new.java\0",
    }
    (vendor / "new.java").write_text("new source")
    monkeypatch.setattr(
        benchmark.subprocess, "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=0, stdout=git_values[tuple(command[3:])]
        ),
    )
    verified_calls = []

    def verify(path):
        verified_calls.append(path)
        return {"patch_id": "late-hero", "verified": True}

    monkeypatch.setattr(benchmark, "verified_installed_backend", verify)
    missing = benchmark.backend_fingerprints()
    assert missing["build_receipt_sha256"] is None
    assert missing["verified_installed_backend"] is None
    assert verified_calls == []
    receipt = install / "cr_rl_backend.json"
    receipt.write_text('{"patch_id":"late-hero"}')
    before = benchmark.backend_fingerprints()
    assert verified_calls == [install]
    assert before["vendor_head"] == "7ee59b39"
    assert before["vendor_status_porcelain"] == [" M source.java"]
    assert len(before["installed_jars_sha256"]) == 6
    assert before["vendor_untracked_sha256"]["new.java"]
    assert before["verified_installed_backend"]["patch_id"] == "late-hero"
    (library / "core.jar").write_text("changed runtime binary")
    (vendor / "new.java").write_text("changed source")
    after = benchmark.backend_fingerprints()
    assert before["installed_jars_sha256"] != after["installed_jars_sha256"]
    assert before["vendor_untracked_sha256"] != after["vendor_untracked_sha256"]

    def reject_receipt(path):
        raise ValueError("Installed JAR differs from build receipt")

    monkeypatch.setattr(benchmark, "verified_installed_backend", reject_receipt)
    invalid = benchmark.backend_fingerprints()
    assert invalid["verified_installed_backend"] is None
    assert "JAR differs" in invalid["backend_validation_error"]
    assert invalid["installed_jars_sha256"] == after["installed_jars_sha256"]
