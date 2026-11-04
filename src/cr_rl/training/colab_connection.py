"""Maintain a real connection to one existing Colab assignment and kernel."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelIdentity:
    endpoint: str
    kernel_id: str
    token_expires_at: float


def watch_connection(
    adapter,
    *,
    budget_seconds=4 * 3600,
    poll_seconds=30,
    refresh_seconds=300,
    rotate_seconds=45 * 60,
    missing_limit=3,
    failure_limit=3,
    transport_errors=(ConnectionError, TimeoutError, OSError),
    clock=time.monotonic,
    wall_time=time.time,
    sleep=time.sleep,
    emit=print,
):
    """Refresh tokens and rotate sockets without executing or restarting work.

    Adapter operations cannot allocate a VM, create a kernel or start a trainer.
    Missing assignments require repeated authoritative reads; network failures
    are errors, never an instruction to allocate another runtime.
    """
    if (
        min(
            budget_seconds,
            poll_seconds,
            refresh_seconds,
            rotate_seconds,
            missing_limit,
            failure_limit,
        )
        <= 0
    ):
        raise ValueError("Watch limits must be positive")
    began = clock()
    expected = None
    connection = None
    renew_at = refresh_at = began
    missing = reconnects = failures = 0
    polls = 0
    try:
        while clock() - began < budget_seconds:
            now = clock()
            needs_connection = connection is None or not adapter.connected(connection)
            try:
                if now >= refresh_at or now >= renew_at or needs_connection:
                    identity = adapter.refresh()
                    if identity is None:
                        missing += 1
                        emit({"phase": "assignment_absent", "consecutive_reads": missing})
                        if missing >= missing_limit:
                            return {"phase": "assignment_gone", "reconnects": reconnects}
                        sleep(poll_seconds)
                        continue
                    missing = 0
                    if expected is None:
                        expected = identity
                    elif (identity.endpoint, identity.kernel_id) != (
                        expected.endpoint,
                        expected.kernel_id,
                    ):
                        raise ValueError(
                            "Assignment or kernel identity changed; refusing to attach"
                        )
                    refresh_at = now + refresh_seconds
                    if needs_connection or now >= renew_at:
                        replacement = adapter.connect(identity)
                        old, connection = connection, replacement
                        # Establish the replacement before closing the old socket.
                        if old is not None:
                            adapter.close(old)
                            reconnects += 1
                        expires_in = identity.token_expires_at - wall_time() - 300
                        if expires_in <= 0:
                            raise ValueError("Refreshed proxy token has insufficient lifetime")
                        renew_at = now + min(rotate_seconds, expires_in)
                        emit(
                            {
                                "phase": "connection_attached"
                                if old is None
                                else "connection_renewed",
                                "kernel_id": identity.kernel_id,
                                "reconnects": reconnects,
                            }
                        )
                state = adapter.inspect(connection)
                polls += 1
                failures = 0
                emit(
                    {
                        "phase": "connection_healthy",
                        "kernel_state": state,
                        "elapsed_seconds": clock() - began,
                        "polls": polls,
                        "reconnects": reconnects,
                    }
                )
            except transport_errors as error:
                failures += 1
                emit(
                    {
                        "phase": "transport_retry",
                        "error_type": type(error).__name__,
                        "consecutive_failures": failures,
                    }
                )
                if failures >= failure_limit:
                    raise ConnectionError(
                        "Repeated transport failure; existing work was not restarted"
                    ) from None
                refresh_at = renew_at = clock()
            sleep(poll_seconds)
        return {
            "phase": "watch_budget_elapsed",
            "polls": polls,
            "reconnects": reconnects,
            "elapsed_seconds": clock() - began,
        }
    finally:
        if connection is not None:
            adapter.close(connection)
