"""Secret-backed incremental Drive sync for Colab recovery."""
from __future__ import annotations
import argparse, base64, os, shutil, subprocess, tempfile
from pathlib import Path

def secret_config() -> bytes:
    try:
        from google.colab import userdata
        value = userdata.get("RCLONE_CONFIG_B64")
    except Exception as exc:
        raise RuntimeError("Colab Secret RCLONE_CONFIG_B64 is unavailable") from exc
    if not value:
        raise RuntimeError("Set Colab Secret RCLONE_CONFIG_B64 before recovery")
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise RuntimeError("RCLONE_CONFIG_B64 is not valid base64") from exc
    if b"[" not in raw or b"type = drive" not in raw:
        raise RuntimeError("The rclone config has no Google Drive remote")
    return raw

def run(remote: str, local: Path, direction: str, dry_run: bool) -> None:
    if not remote.startswith("gdrive:") or direction not in {"pull", "push"}:
        raise ValueError("Only gdrive: pull/push is accepted")
    local.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crrl-rclone-") as td:
        config = Path(td) / "rclone.conf"
        config.write_bytes(secret_config()); os.chmod(config, 0o600)
        source, destination = (remote, str(local)) if direction == "pull" else (str(local), remote)
        command = ["rclone", "copy", source, destination, "--config", str(config), "--immutable",
                   "--checkers", "4", "--transfers", "2", "--stats-one-line", "--stats", "30s"]
        if dry_run: command.append("--dry-run")
        subprocess.run(command, check=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=("pull", "push")); parser.add_argument("remote")
    parser.add_argument("local", type=Path); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if shutil.which("rclone") is None: raise SystemExit("rclone is not installed")
    run(args.remote.rstrip("/"), args.local, args.direction, args.dry_run)

if __name__ == "__main__": main()
