import sys
from types import SimpleNamespace

import pytest

from cr_rl.training.colab_connection import KernelIdentity
from scripts.watch_colab_session import ExistingColab


def adapter():
    result = ExistingColab.__new__(ExistingColab)
    result.endpoint, result.kernel_id = "fixed", "existing-kernel"
    result.session = SimpleNamespace(url="https://example.invalid", token="test-only")
    return result


def install_client(monkeypatch, has_kernel):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))
            self._manager = SimpleNamespace(
                has_kernel=has_kernel,
                client=SimpleNamespace(connection_ready=SimpleNamespace(wait=lambda _: True)),
            )

        def start(self, **kwargs):
            calls.append(("start", kwargs))

        def stop(self, **kwargs):
            calls.append(("stop", kwargs))

    monkeypatch.setitem(
        sys.modules,
        "jupyter_kernel_client",
        SimpleNamespace(
            JupyterKernelClient=Client,
            JupyterSubprotocol=SimpleNamespace(DEFAULT="default"),
        ),
    )
    return calls


def test_missing_kernel_cannot_trigger_sdk_implicit_creation(monkeypatch):
    calls = install_client(monkeypatch, has_kernel=False)
    with pytest.raises(ValueError, match="refusing implicit kernel creation"):
        adapter().connect(KernelIdentity("fixed", "existing-kernel", 9999))
    assert [name for name, _ in calls] == ["construct", "stop"]
    assert calls[-1][1] == {"shutdown_kernel": False}


def test_existing_kernel_uses_distinct_client_and_never_shuts_down_kernel(monkeypatch):
    calls = install_client(monkeypatch, has_kernel=True)
    connection = adapter().connect(KernelIdentity("fixed", "existing-kernel", 9999))
    assert connection._own_kernel is False
    args = calls[0][1]
    assert args["kernel_id"] == "existing-kernel"
    assert "session" not in args["client_kwargs"]
    ExistingColab.close(connection)
    assert calls[-1] == ("stop", {"shutdown_kernel": False})


def test_refresh_does_not_attach_to_a_new_assignment_with_same_session_name():
    result = adapter()
    result.state = SimpleNamespace(
        client=SimpleNamespace(
            list_assignments=lambda: [SimpleNamespace(endpoint="different-endpoint")],
        )
    )
    assert result.refresh() is None
