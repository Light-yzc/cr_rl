import importlib.util
import json
from pathlib import Path

import pytest

from cr_rl.training.persistence import file_sha256


def test_release_requires_final_weight_pool_report_and_all_native_replay_hashes(
    tmp_path, monkeypatch
):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "continued_release", scripts / "release_continued_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    summary = write(
        "learning_curve_summary.json",
        dict(comparisons={str(i): dict(games=16, native_parity_games=16) for i in range(4)}),
    )
    write("evaluation_status.json", dict(phase="completed", summary_sha256=file_sha256(summary)))
    pool = write("pool.json", dict(entries=[]))
    checkpoint = write("final.pt", dict(fixture="test-only"))
    write(
        "latest.json",
        dict(
            update=14,
            pool="pool.json",
            pool_sha256=file_sha256(pool),
            files=[
                dict(
                    path="final.pt", bytes=checkpoint.stat().st_size, sha256=file_sha256(checkpoint)
                )
            ],
        ),
    )
    for chunk in range(8):
        directory = f"learning_curve/comparison{chunk}/pairs_16"
        report = write(f"{directory}/evaluation.json", dict(games=8))
        games = []
        for index in range(8):
            replay = write(f"{directory}/match_{index:03d}.json", dict(index=index))
            games.append(dict(native_outcome_parity=True, replay_sha256=file_sha256(replay)))
        write(
            f"{directory}/execution_audit.json",
            dict(evaluation_sha256=file_sha256(report), games=games),
        )
    assert module.verify_completion(tmp_path)["games"] == 64
    write("learning_curve/comparison7/pairs_16/match_007.json", dict(corrupted=True))
    with pytest.raises(ValueError, match="replay integrity"):
        module.verify_completion(tmp_path)
    write("evaluation_status.json", dict(phase="evaluating"))
    with pytest.raises(ValueError, match="unfinished"):
        module.verify_completion(tmp_path)
