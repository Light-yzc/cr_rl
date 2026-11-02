#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
command -v brew >/dev/null 2>&1 || { echo "Install Homebrew from https://brew.sh/" >&2; exit 1; }
brew list --versions openjdk@21 >/dev/null 2>&1 || brew install openjdk@21
JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home"
if [[ ! -x "$JAVA_HOME/bin/java" ]]; then JAVA_HOME="$(/usr/libexec/java_home -v 21)"; fi
export JAVA_HOME PATH="$JAVA_HOME/bin:$PATH"
java -version
command -v uv >/dev/null 2>&1 || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
uv python install 3.12
uv sync --python 3.12 --group dev
VENDOR="$ROOT/.cache/vendor/crforge"
mkdir -p "$(dirname "$VENDOR")"
if [[ ! -d "$VENDOR/.git" ]]; then git clone https://github.com/voonhous/crforge.git "$VENDOR"; fi
git -C "$VENDOR" fetch --depth 1 origin 7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6
git -C "$VENDOR" checkout --detach 7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6
cd "$VENDOR"
./gradlew :env:installDist :tables:dataVersionTables -Pcrforge.assetSource=cdn -Pcrforge.dataVersion=16.635.3
cd "$ROOT"
uv pip install --python .venv/bin/python -e "$VENDOR/python"
echo "Bootstrap complete. Run: uv run python -m cr_rl.cli doctor"
