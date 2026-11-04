import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from test_repeated_evaluation import fixtures

spec = importlib.util.spec_from_file_location(
    "replication_resume", Path(__file__).parents[1] / "scripts/colab_seed_replication.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def saved_chunk(tmp_path):
    protocol, names, results = fixtures()
    protocol["evaluation_code_sha256"] = "code"
    result = results[0]
    result.update(evaluation_code_sha256="code", evaluation_code_file_sha256_at_end="code")
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(result))
    games = []
    for i, match in enumerate(result["matchups"]):
        replay = tmp_path / f"match_{i:03d}.json"
        replay.write_text("{}")
        games.append(
            {
                **{k: match[k] for k in ["pair", "learner_seat", "reward"]},
                "native_outcome_parity": True,
                "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
            }
        )
    audit = dict(evaluation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), games=games)
    (tmp_path / "execution_audit.json").write_text(json.dumps(audit))
    return protocol, names, protocol["blocks"][0], 0


def test_resume_accepts_only_the_registered_complete_replayed_chunk(tmp_path):
    args = saved_chunk(tmp_path)
    assert module.verify_saved_chunk(tmp_path, *args)["games"] == 2


@pytest.mark.parametrize("change", ["replay", "missing_audit", "identity", "code", "weights"])
def test_resume_rejects_wrong_or_incomplete_backups(tmp_path, change):
    protocol, names, block, start = saved_chunk(tmp_path)
    if change == "replay":
        (tmp_path / "match_000.json").write_text("changed")
    elif change == "missing_audit":
        (tmp_path / "execution_audit.json").unlink()
    elif change == "identity":
        path = tmp_path / "execution_audit.json"
        audit = json.loads(path.read_text())
        audit["games"][0]["learner_seat"] = 1
        path.write_text(json.dumps(audit))
    elif change == "code":
        protocol["evaluation_code_sha256"] = "wrong"
    else:
        protocol["policy_sha256"] = "wrong"
    with pytest.raises(ValueError):
        module.verify_saved_chunk(tmp_path, protocol, names, block, start)
