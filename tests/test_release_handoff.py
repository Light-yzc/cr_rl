import importlib.util
from pathlib import Path


def test_release_handoff_requires_two_live_project_processes(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "release_handoff", scripts / "release_continued_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = tmp_path / "current"
    target = tmp_path / "next"
    request = dict(backup=str(target), controller_pid=1, release_guard_pid=2)

    def alive(pid):
        return pid in (1, 2)

    def command(pid):
        return f"python scripts/worker.py --backup {target}".encode()

    assert module.live_handoff(request, current, alive, command)
    assert not module.live_handoff(dict(request, release_guard_pid=1), current, alive, command)
    assert not module.live_handoff(dict(request, release_guard_pid=3), current, alive, command)
    assert not module.live_handoff(dict(request, backup=str(current)), current, alive, command)
    assert not module.live_handoff(request, current, alive, lambda pid: b"unrelated")
    assert not module.live_handoff(
        dict(request, backup=str(tmp_path.parent / "other")), current, alive, command
    )
