import importlib.util
from pathlib import Path

import pytest


def test_continuation_preserves_training_hyperparameters_and_original_command(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "ppo_continue", Path(__file__).parents[1] / "scripts/continue_ppo_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = [
        "python",
        "scripts/train_selfplay.py",
        "--resume",
        "old.pt",
        "--updates",
        "3",
        "--out",
        "old",
        "--backup-dir",
        "old-drive",
        "--progress-jsonl",
        "old-log",
        "--batch-size",
        "8",
        "--grad-accumulation",
        "8",
        "--gamma",
        "0.9995",
    ]
    before = original.copy()
    result = module.continuation_command(original, tmp_path / "run", tmp_path / "backup", 8)
    assert original == before
    assert result[result.index("--updates") + 1] == "8"
    assert result[result.index("--resume") + 1] == str(tmp_path / "run/ppo_latest.pt")
    assert result[result.index("--batch-size") :] == original[original.index("--batch-size") :]
    with pytest.raises(ValueError):
        module.continuation_command(original, tmp_path, tmp_path, 0)
