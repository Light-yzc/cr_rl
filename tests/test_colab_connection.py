import pytest

from cr_rl.training.colab_connection import KernelIdentity, watch_connection


class Adapter:
    def __init__(self):
        self.now = 0
        self.events = []
        self.number = 0
        self.missing_at = set()
        self.error_at = set()
        self.drop_at = set()
        self.changed_at = None

    def refresh(self):
        self.events.append((self.now, "refresh"))
        if self.now in self.missing_at:
            return None
        return KernelIdentity(
            "other" if self.changed_at == self.now else "fixed", "kernel", self.now + 3600
        )

    def connect(self, identity):
        self.number += 1
        self.events.append((self.now, "open", self.number))
        return self.number

    def connected(self, connection):
        return self.now not in self.drop_at

    def inspect(self, connection):
        if self.now in self.error_at:
            raise ConnectionError("Simulated network outage")
        return "busy"

    def close(self, connection):
        self.events.append((self.now, "close", connection))

    def sleep(self, seconds):
        self.now += seconds


def watch(adapter, **kwargs):
    return watch_connection(
        adapter,
        budget_seconds=6,
        poll_seconds=1,
        refresh_seconds=1,
        clock=lambda: adapter.now,
        wall_time=lambda: adapter.now,
        sleep=adapter.sleep,
        emit=lambda _: None,
        **kwargs,
    )


def test_token_rotation_connects_before_closing_without_restarting_work():
    adapter = Adapter()
    result = watch(adapter, rotate_seconds=2)
    assert result["phase"] == "watch_budget_elapsed"
    assert result["reconnects"] == 2
    assert adapter.events.index((2, "open", 2)) < adapter.events.index((2, "close", 1))
    assert adapter.events[-1] == (6, "close", 3)


def test_transport_drop_reattaches_same_kernel():
    adapter = Adapter()
    adapter.drop_at = {1}
    result = watch(adapter)
    assert result["reconnects"] == 1 and adapter.number == 2


def test_one_missing_assignment_response_does_not_stop_or_allocate():
    adapter = Adapter()
    adapter.missing_at = {2}
    assert watch(adapter)["phase"] == "watch_budget_elapsed"
    assert adapter.number == 1


def test_repeated_authoritative_absence_stops_without_auto_allocation():
    adapter = Adapter()
    adapter.missing_at = {1, 2, 3}
    assert watch(adapter)["phase"] == "assignment_gone"
    assert adapter.number == 1


def test_different_assignment_cannot_be_mistaken_for_reconnection():
    adapter = Adapter()
    adapter.changed_at = 2
    with pytest.raises(ValueError, match="identity changed"):
        watch(adapter)
    assert adapter.number == 1


def test_network_error_does_not_prove_assignment_deletion():
    adapter = Adapter()
    adapter.error_at = {0, 1, 2}
    with pytest.raises(ConnectionError, match="existing work was not restarted"):
        watch(adapter)
    assert adapter.number == 3


def test_configurable_network_retry_budget_survives_three_transient_errors():
    adapter = Adapter()
    adapter.error_at = {0, 1, 2}
    result = watch(adapter, failure_limit=5)
    assert result["phase"] == "watch_budget_elapsed"
    assert result["polls"] == 3
    assert adapter.number == 4


def test_explicit_sdk_transport_error_is_retryable_without_logging_secrets():
    class SDKTransportError(Exception):
        pass

    class RefreshFailure(Adapter):
        def refresh(self):
            if self.now in {0, 1}:
                raise SDKTransportError("secret-bearing transport URL")
            return super().refresh()

    adapter = RefreshFailure()
    result = watch(adapter, transport_errors=(ConnectionError, SDKTransportError))
    assert result["phase"] == "watch_budget_elapsed"
    assert adapter.number == 1
