import json

import pytest

from scripts.run_bc_repair_trial import resume_command, trim_uncheckpointed_log


def test_resume_keeps_protocol_but_trains_only_remaining_updates():
    original = [
        "python", "scripts/train_bc.py", "data", "--init-from-bc", "parent.pt",
        "--carry-optimizer", "--upgrade-readouts", "--max-updates", "1000",
        "--timing-objective", "natural", "--shard-mix-window", "64",
    ]
    result = resume_command(original, "repaired.pt", 4864, 5096)
    assert result[result.index("--resume") + 1] == "repaired.pt"
    assert result[result.index("--max-updates") + 1] == "232"
    assert "--carry-optimizer" not in result and "--upgrade-readouts" not in result
    assert result[-4:] == original[-4:]
    assert "--init-from-bc" in original
    with pytest.raises(ValueError, match="unfinished"):
        resume_command(original, "repaired.pt", 5096, 5096)


def test_resumed_logs_exclude_lost_updates(tmp_path):
    path = tmp_path / "train.log"
    path.write_text("setup\n" + "\n".join(
        json.dumps({"optimizer_steps": step}) for step in (4832, 4864, 4896, 4944)
    ) + "\n")
    trim_uncheckpointed_log(path, 4864)
    assert path.read_text().splitlines() == [
        "setup", '{"optimizer_steps": 4832}', '{"optimizer_steps": 4864}',
    ]
