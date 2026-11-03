#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Install a JDK with the system package manager first, e.g. on Ubuntu:
# sudo apt-get update && sudo apt-get install -y openjdk-21-jdk-headless git
command -v java >/dev/null || { echo 'Install Java 21 first.' >&2; exit 1; }
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")}"
export PATH="$JAVA_HOME/bin:$PATH"
command -v uv >/dev/null || { echo 'Install uv first (python -m pip install uv).' >&2; exit 1; }
uv sync --python 3.12 --group dev --frozen
uv run python scripts/setup_simulator.py
uv run python -m cr_rl.cli doctor
