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
    seeded = original + ["--history-seed-manifest", "first_launch_only.json", "--role-league"]
    resumed = module.continuation_command(seeded, tmp_path / "next", tmp_path / "drive", 8)
    assert "--history-seed-manifest" not in resumed and "first_launch_only.json" not in resumed
    assert "--role-league" in resumed and seeded[-3:] == [
        "--history-seed-manifest",
        "first_launch_only.json",
        "--role-league",
    ]
    with pytest.raises(ValueError):
        module.continuation_command(original, tmp_path, tmp_path, 0)
    with pytest.raises(ValueError, match="require v2"):
        module.continuation_command(
            original, tmp_path, tmp_path, 2, actor_effective_batch_size=256
        )
    v2 = original + ["--v2"]
    changed = module.continuation_command(
        v2, tmp_path, tmp_path, 2, actor_effective_batch_size=256
    )
    assert "--actor-effective-batch-size" not in v2
    assert changed[-2:] == ["--actor-effective-batch-size", "256"]
    assert changed[changed.index("--batch-size") + 1] == "8"
    assert changed[changed.index("--grad-accumulation") + 1] == "8"
    changed = module.continuation_command(
        changed, tmp_path, tmp_path, 2, actor_effective_batch_size=128
    )
    assert changed.count("--actor-effective-batch-size") == 1
    assert changed[-2:] == ["--actor-effective-batch-size", "128"]
    for value in (0, 32, 65):
        with pytest.raises(ValueError):
            module.continuation_command(
                v2, tmp_path, tmp_path, 2, actor_effective_batch_size=value
            )
    curriculum = [*v2, "--expand-decks", "six.yaml", "--decks", "old.yaml"]
    resumed = module.continuation_command(curriculum, tmp_path, tmp_path, 2)
    assert "--expand-decks" not in resumed
    assert resumed[resumed.index("--decks") + 1] == "six.yaml"
    assert curriculum[-4:] == ["--expand-decks", "six.yaml", "--decks", "old.yaml"]
