"""Rebuild the full packed-data Colab notebook with verified wrapper modules.

The source template deliberately has no outputs, credentials, or embedded binary
tools. Keeping it separately makes changes to training settings reviewable.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts/templates/colab_bc040_full.json"
TARGET = ROOT / "notebooks/colab_bc040_full.ipynb"
HELPERS = ("notebook_state_bc.py", "notebook_state_bc_setup.py", "notebook_state_bc_launch.py")
PLACEHOLDER = "__EMBEDDED_MODULES__"


def build_full():
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    modules = {}
    for name in HELPERS:
        raw = (ROOT / "scripts" / name).read_bytes()
        modules[name] = {
            "base64": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    embedded = json.dumps(modules, separators=(",", ":"))
    replacements = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        replacements += source.count(PLACEHOLDER)
        source = source.replace(PLACEHOLDER, embedded)
        cell["source"] = source.splitlines(keepends=True)
        cell["metadata"] = {}
        cell["id"] = hashlib.sha256((cell["cell_type"] + source).encode()).hexdigest()[:12]
        if cell["cell_type"] == "code":
            compile(source, "<colab_bc040_full>", "exec")
            cell["execution_count"] = None
            cell["outputs"] = []
    if replacements != 1:
        raise ValueError("The full notebook template must embed tools exactly once")
    notebook["metadata"] = {
        "accelerator": "GPU",
        "colab": {"name": TARGET.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return notebook


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(build_full(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
