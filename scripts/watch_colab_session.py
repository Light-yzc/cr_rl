"""Hold and refresh an existing Colab kernel connection; never allocate or execute.

Run with the Python interpreter of the installed google-colab-cli environment.
The host running this command must remain awake and online. Start the real
kernel-side trial supervisor separately before attaching this watcher.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cr_rl.training.colab_connection import KernelIdentity, watch_connection  # noqa: E402


class ExistingColab:
    def __init__(self, name):
        from colab_cli.common import state

        self.state = state
        self.session = state.store.get(name)
        if self.session is None or not self.session.kernel_id:
            raise ValueError(
                "An existing registered session and kernel are required; no auto-create"
            )
        self.endpoint = self.session.endpoint
        self.kernel_id = self.session.kernel_id

    def refresh(self):
        # Do not use resolve_session/ssh: those paths can auto-allocate a VM.
        assignments = self.state.client.list_assignments()
        match = next((row for row in assignments if row.endpoint == self.endpoint), None)
        if match is None:
            return None
        info = match.runtime_proxy_info
        self.session.token, self.session.url = info.token, info.url
        self.session.token_expires_at = info.expires_at()
        self.state.store.add(self.session)
        return KernelIdentity(
            self.endpoint, self.kernel_id, self.session.token_expires_at.timestamp()
        )

    def connect(self, identity):
        import jupyter_kernel_client as jkc

        if identity.kernel_id != self.kernel_id or identity.endpoint != self.endpoint:
            raise ValueError("Refusing a different session/kernel")
        if hasattr(jkc, "ColabKernelClient"):
            cls, token_key = jkc.ColabKernelClient, "proxy_token"
        elif hasattr(jkc, "KernelClient"):
            cls, token_key = jkc.KernelClient, "token"
        else:
            cls, token_key = jkc.JupyterKernelClient, "token"
        client = cls(
            server_url=self.session.url,
            kernel_id=self.kernel_id,
            **{token_key: self.session.token},
            client_kwargs={
                "subprotocol": jkc.JupyterSubprotocol.DEFAULT,
                "extra_params": {"colab-runtime-proxy-token": self.session.token},
            },
            headers={
                "X-Colab-Client-Agent": "colab-cli",
                "X-Colab-Runtime-Proxy-Token": self.session.token,
            },
        )
        client._own_kernel = False
        if not client._manager.has_kernel:
            client.stop(shutdown_kernel=False)
            raise ValueError("Existing kernel missing; refusing implicit kernel creation")
        try:
            client.start(timeout=20)
            if not client._manager.client.connection_ready.wait(20):
                raise TimeoutError("Kernel connection handshake timed out")
        except Exception:
            client.stop(shutdown_kernel=False)
            raise
        return client

    @staticmethod
    def connected(client):
        return client._manager.client.connection_ready.is_set()

    @staticmethod
    def inspect(client):
        # This observer does not execute code or retain another client's outputs.
        client._manager.client.iopub_channel.get_msgs()
        model = client._manager.refresh_model(timeout=20)
        if model is None:
            raise ValueError("Kernel missing but assignment may exist; inspect before restarting")
        return model.get("execution_state", "unknown")

    @staticmethod
    def close(client):
        client.stop(shutdown_kernel=False)


def main():
    from google.auth.exceptions import TransportError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--minutes", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--refresh-seconds", type=float, default=300)
    parser.add_argument("--rotate-seconds", type=float, default=2700)
    parser.add_argument("--failure-limit", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Never log HTTP headers, proxy tokens, auth responses or kernel outputs.
    logging.disable(logging.CRITICAL)
    with args.out.open("a") as log:

        def emit(event):
            text = json.dumps({"observed_unix": time.time(), **event})
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()

        try:
            result = watch_connection(
                ExistingColab(args.session),
                budget_seconds=args.minutes * 60,
                poll_seconds=args.poll_seconds,
                refresh_seconds=args.refresh_seconds,
                rotate_seconds=args.rotate_seconds,
                failure_limit=args.failure_limit,
                transport_errors=(ConnectionError, TimeoutError, OSError, TransportError),
                emit=emit,
            )
            emit(result)
            if result["phase"] == "assignment_gone":
                raise SystemExit(2)
        except Exception as error:
            # Raw transport exception text can contain token-bearing URLs.
            emit({"phase": "watch_failed", "error_type": type(error).__name__})
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
