"""Notebook-only logging/progress contracts; no Torch, W&B or cloud needed."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("notebook_state_bc", Path(__file__).resolve().parents[1] / "scripts/notebook_state_bc.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def train(index, step, epoch=1):
    return {"phase": "state_bc_train", "epoch": epoch, "episode_index": index,
            "optimizer_steps": step, "loss": 6.1, "amp_overflow_skipped": False}


def test_tailer_does_not_lose_partial_append_or_replay_acknowledged_events(tmp_path):
    log, cursor = tmp_path / "metrics.jsonl", tmp_path / "cursor.json"
    first = json.dumps(train(0, 257)).encode() + b"\n"
    second = json.dumps(train(1, 258)).encode() + b"\n"
    log.write_bytes(first + second[:20])
    tail = helper.JSONLTailer(log, cursor, identity="run-a")
    pending = list(tail.pending())
    assert [row[0]["optimizer_steps"] for row in pending] == [257]
    tail.acknowledge(pending[0][1])
    assert list(tail.pending()) == []
    with log.open("ab") as out:
        out.write(second[20:])
    restored = helper.JSONLTailer(log, cursor, identity="run-a")
    pending = list(restored.pending())
    assert [row[0]["optimizer_steps"] for row in pending] == [258]
    restored.acknowledge(pending[0][1])
    assert list(helper.JSONLTailer(log, cursor, identity="run-a").pending()) == []


def test_failed_delivery_keeps_event_pending_and_other_run_replays(tmp_path):
    log, cursor = tmp_path / "metrics.jsonl", tmp_path / "cursor.json"
    log.write_text(json.dumps(train(0, 257)) + "\n")
    tail = helper.JSONLTailer(log, cursor, identity="run-a")
    row, end = next(tail.pending())
    assert next(tail.pending())[0] == row
    tail.acknowledge(end)
    assert list(helper.JSONLTailer(log, cursor, identity="run-a").pending()) == []
    assert len(list(helper.JSONLTailer(log, cursor, identity="run-b").pending())) == 1


def test_tailer_detects_truncated_and_same_size_replaced_logs(tmp_path):
    log, cursor = tmp_path / "metrics.jsonl", tmp_path / "cursor.json"
    log.write_text(json.dumps(train(0, 257)) + "\n" + json.dumps(train(1, 258)) + "\n")
    tail = helper.JSONLTailer(log, cursor, identity="run")
    for row, offset in tail.pending():
        tail.acknowledge(offset)
    log.write_text(json.dumps(train(0, 259)) + "\n")
    with pytest.warns(RuntimeWarning, match="truncated/replaced"):
        rows = list(tail.pending())
    assert rows[0][0]["optimizer_steps"] == 259
    tail.acknowledge(rows[0][1])
    log.write_text(json.dumps(train(0, 260)) + "\n")
    with pytest.warns(RuntimeWarning, match="truncated/replaced"):
        assert next(tail.pending())[0]["optimizer_steps"] == 260


def test_trajectory_progress_uses_seats_instead_of_global_optimizer_offset():
    assert helper.resume_completed({"epoch": 0, "next_episode": 96, "optimizer_steps": 352}, 350) == 96
    assert helper.resume_completed({"epoch": 1, "next_episode": 0}, 350) == 350
    assert helper.completed_seats(train(0, 257), 350) == 1
    assert helper.completed_seats(train(349, 606), 350) == 350
    assert helper.completed_seats(train(0, 607, epoch=2), 350) == 351
    assert helper.completed_seats({"phase": "state_bc_validation", "epoch": 1, "optimizer_steps": 256}, 350) is None
    with pytest.raises(ValueError):
        helper.resume_completed({"epoch": 0, "next_episode": 351}, 350)


def test_metric_namespaces_retain_validation_and_train_at_same_step():
    a = helper.metric_payload(train(0, 257))
    b = helper.metric_payload({"phase": "state_bc_validation", "epoch": 1, "optimizer_steps": 257,
                               "loss": 6.2, "act_recall": 0.2, "aggregation": "seat", "bad": float("nan")})
    assert a["trainer/optimizer_steps"] == b["trainer/optimizer_steps"] == 257
    assert a["train/loss"] == 6.1
    assert b["validation/loss"] == 6.2
    assert "validation/bad" not in b and "validation/aggregation" not in b
    assert a["train/amp_overflow_skipped"] == 0


class Bar:
    def __init__(self, n=0):
        self.n, self.postfix, self.closed = n, {}, False
    def update(self, amount):
        self.n += amount
    def set_postfix(self, value):
        self.postfix = value
    def close(self):
        self.closed = True


class Run:
    id = "test-run"
    def __init__(self, error=False):
        self.logged, self.error = [], error
    def log(self, payload):
        if self.error:
            raise ConnectionError("W&B unavailable")
        self.logged.append(payload)


class Process:
    def __init__(self, code=0):
        self.code = code
    def poll(self):
        return self.code


def test_monitor_replays_metrics_and_resumes_progress_without_fake_global_step(tmp_path):
    log, trainer_log, cursor = tmp_path / "metrics.jsonl", tmp_path / "training.log", tmp_path / "cursor.json"
    rows = [train(0, 257), train(1, 258), {"phase": "state_bc_validation", "epoch": 1,
            "optimizer_steps": 258, "loss": 6.0}]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trainer_log.write_text("ok")
    run, bar = Run(), Bar(n=1)
    result = helper.monitor_process(Process(), log, trainer_log, run=run, target_epochs=1,
        seats_per_epoch=2, initial_completed=1, cursor_path=cursor, bar=bar)
    assert result["completed_seats"] == 2 and bar.n == 2 and bar.closed
    assert len(run.logged) == 3
    assert run.logged[-1]["validation/loss"] == 6.0
    assert helper.monitor_process(Process(), log, trainer_log, run=run, target_epochs=1,
        seats_per_epoch=2, initial_completed=2, cursor_path=cursor, bar=Bar(2))["completed_seats"] == 2
    assert len(run.logged) == 3


def test_failed_wandb_delivery_is_visible_and_does_not_acknowledge(tmp_path):
    log, cursor = tmp_path / "metrics.jsonl", tmp_path / "cursor.json"
    log.write_text(json.dumps(train(0, 257)) + "\n")
    result = helper.monitor_process(Process(), log, tmp_path / "training.log", run=Run(error=True),
        target_epochs=1, seats_per_epoch=2, cursor_path=cursor, bar=Bar())
    assert result['logging_status'] == 'pending'
    assert result['logging_error_type'] == 'ConnectionError'
    assert not cursor.exists()


def test_nonzero_trainer_return_code_surfaces_log_tail(tmp_path):
    trainer_log = tmp_path / "training.log"
    trainer_log.write_text("Traceback: actual trainer failure")
    bar = Bar()
    with pytest.raises(RuntimeError, match="code 2.*", ) as caught:
        helper.monitor_process(Process(2), tmp_path / "metrics.jsonl", trainer_log,
            target_epochs=1, seats_per_epoch=2, bar=bar)
    assert "actual trainer failure" in str(caught.value)
    assert bar.closed


def test_progress_continues_during_wandb_failure_and_pending_rows_retry(tmp_path, monkeypatch):
    log, cursor = tmp_path / "metrics.jsonl", tmp_path / "cursor.json"
    log.write_text(json.dumps(train(0, 257)) + "\n" + json.dumps(train(1, 258)) + "\n")
    run, bar = Run(error=True), Bar()
    class LiveProcess:
        calls = 0
        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 0
    def recover(_):
        assert bar.n == 2  # Display did not wait for a functioning upload.
        run.error = False
    monkeypatch.setattr(helper.time, "sleep", recover)
    result = helper.monitor_process(LiveProcess(), log, tmp_path / "training.log", run=run,
        target_epochs=1, seats_per_epoch=2, cursor_path=cursor, bar=bar)
    assert result['logging_status'] == 'accepted'
    assert [event['trainer/optimizer_steps'] for event in run.logged] == [257, 258]
    assert result['completed_seats'] == 2


def test_interrupt_closes_monitor_without_terminating_owned_process(tmp_path, capsys):
    class InterruptedProcess:
        def poll(self):
            raise KeyboardInterrupt
    bar = Bar()
    with pytest.raises(KeyboardInterrupt):
        helper.monitor_process(InterruptedProcess(), tmp_path / 'metrics.jsonl', tmp_path / 'training.log',
            target_epochs=1, seats_per_epoch=2, bar=bar)
    assert bar.closed
    assert 'trainer未被本监控停止' in capsys.readouterr().out


def test_prepare_log_skips_plain_stderr_but_metrics_remain_strict(tmp_path):
    path=tmp_path/'training.log'
    path.write_text('Traceback: old controller stopped\n  File "trainer.py", line 1\n'+json.dumps({'phase':'state_bc_prepare'})+'\n')
    with pytest.raises(ValueError,match='Invalid complete'):
        list(helper.JSONLTailer(path).pending())
    tail=helper.JSONLTailer(path,skip_non_json=True)
    records=list(tail.pending())
    assert len(records)==1 and records[0][0]['phase']=='state_bc_prepare'
    tail.acknowledge(records[0][1])
    assert list(tail.pending())==[]


def test_window_metrics_have_a_clean_wandb_series_without_relabelling_old_points():
    row={'phase':'state_bc_train','optimizer_steps':32,'slot_accuracy':.6,
         'play_rows':1000,'aggregation':'optimizer_update_window_v1'}
    result=helper.metric_payload(row)
    assert result['train_window/slot_accuracy']==.6
    assert result['train_window/play_rows']==1000
    assert result['train/slot_accuracy']==.6
    del row['aggregation']
    assert 'train_window/slot_accuracy' not in helper.metric_payload(row)


def test_monitor_metadata_sync_failure_does_not_stop_training_progress(tmp_path, capsys):
    log = tmp_path / 'metrics.jsonl'
    log.write_text(json.dumps(train(0, 257)) + '\n' + json.dumps(train(1, 258)) + '\n')
    calls = []
    def failed_sync():
        calls.append(1)
        raise OSError('Drive unavailable')
    bar = Bar()
    result = helper.monitor_process(Process(), log, tmp_path / 'training.log',
        target_epochs=1, seats_per_epoch=2, bar=bar, on_poll=failed_sync)
    assert result['completed_seats'] == 2 and bar.n == 2 and bar.closed
    assert len(calls) == 2  # One iteration plus the final drain callback.
    assert result['monitor_sync_errors'] == 2
    assert '本次同步未成功' in capsys.readouterr().out
