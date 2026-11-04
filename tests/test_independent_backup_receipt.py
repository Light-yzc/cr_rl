import importlib.util
import json
from pathlib import Path

import pytest

from cr_rl.training.persistence import file_sha256


def test_independent_receipt_binds_all_files_and_exact_generation(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "backup_receipt", scripts / "release_continued_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_path = tmp_path / "checkpoints/update_000022.manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            dict(checkpoint="model.pt", files=[dict(path="model.pt", sha256="actual")], update=22)
        )
    )
    path = tmp_path / "receipt.json"
    receipt = dict(
        generation=22,
        manifest_sha256=file_sha256(manifest_path),
        checkpoint_sha256="actual",
        backup_files_verified=1,
        full_weight_and_pool_hash_verified=True,
    )
    path.write_text(json.dumps(receipt))
    module.verify_independent_receipt(tmp_path, path, 22)
    for override in (
        dict(generation=21),
        dict(manifest_sha256="old"),
        dict(backup_files_verified=0),
        dict(full_weight_and_pool_hash_verified=False),
        dict(checkpoint_sha256="wrong"),
    ):
        path.write_text(json.dumps(dict(receipt, **override)))
        with pytest.raises(ValueError, match="receipt"):
            module.verify_independent_receipt(tmp_path, path, 22)
