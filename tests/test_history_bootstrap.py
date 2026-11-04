import json
from pathlib import Path

import pytest
import torch

from cr_rl.models import ActorPolicy, ModelConfig
from cr_rl.opponents.bootstrap import seed_history
from cr_rl.opponents.league import OpponentEntry, OpponentPool
from cr_rl.training.persistence import file_sha256


def test_seed_verifies_all_inputs_before_changing_pool(tmp_path):
    config = ModelConfig().__dict__
    source = tmp_path / "actor.pt"
    torch.save(
        dict(
            model=ActorPolicy(ModelConfig()).state_dict(),
            config=config,
            updates=14,
            backend_commit="abc",
        ),
        source,
    )
    row = dict(checkpoint=str(source), generation=14, sha256=file_sha256(source))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"policies": [row, dict(row, generation=10)]}))
    pool = OpponentPool([OpponentEntry(0, "bc.pt", 0, role="anchor")], max_entries=8)
    with pytest.raises(ValueError, match="generation"):
        seed_history(pool, manifest, tmp_path, config, {"backend_commit": "abc"})
    assert len(pool.entries) == 1 and not (tmp_path / "history_bootstrap.json").exists()
    manifest.write_text(json.dumps({"policies": [row]}))
    before = torch.get_rng_state().clone()
    seed_history(pool, manifest, tmp_path, config, {"backend_commit": "abc"})
    assert torch.equal(before, torch.get_rng_state())
    entry = pool.entries[-1]
    assert entry.role == "historical" and entry.policy_id >= (1 << 30) and entry.generation == 14
    assert file_sha256(Path(entry.checkpoint)) == row["sha256"]
    with pytest.raises(ValueError, match="fresh history"):
        seed_history(pool, manifest, tmp_path, config, {"backend_commit": "abc"})
