import sys
from pathlib import Path
from types import ModuleType

import pytest

from scripts import train_selfplay


@pytest.fixture
def cli(monkeypatch):
    """Exercise parser dispatch without creating models, CUDA contexts or simulators."""
    dispatched = []
    runner = ModuleType("cr_rl.training.selfplay_v2")
    runner.run = dispatched.append
    monkeypatch.setitem(sys.modules, runner.__name__, runner)

    def parse(*arguments):
        monkeypatch.setattr(sys, "argv", ["train_selfplay.py", *arguments])
        train_selfplay.main()
        return dispatched[-1]

    return parse


def test_convergence_controls_are_opt_in_and_keep_dynamic_ceiling(cli):
    default = cli("--v2")
    assert default.adaptive_actor_lr is False
    assert default.adaptive_actor_lr_min == 2.5e-6
    assert default.adaptive_actor_lr_max is None
    assert default.league_diversity_manifest is None
    enabled = cli(
        "--v2", "--adaptive-actor-lr",
        "--target-card-kl", "0.015", "--target-tile-kl", "0.06",
    )
    assert enabled.adaptive_actor_lr is True
    assert enabled.adaptive_actor_lr_max is None


@pytest.mark.parametrize("arguments, message", [
    (("--adaptive-actor-lr",), "requires v2"),
    (("--v2", "--adaptive-actor-lr"), "both Card/Tile KL"),
    (("--adaptive-actor-lr-min", "nan"), "minimum must be finite"),
    (("--adaptive-actor-lr-min", "0"), "minimum must be finite"),
    (("--adaptive-actor-lr-max", "inf"), "maximum must be finite"),
    (("--adaptive-actor-lr-max", "1e-7"), "at least the minimum"),
])
def test_invalid_adaptive_controls_fail_before_training(cli, capsys, arguments, message):
    with pytest.raises(SystemExit) as error:
        cli(*arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_manifest_requires_a_file_and_v2(cli, capsys, tmp_path):
    manifest = tmp_path / "diversity.json"
    with pytest.raises(SystemExit):
        cli("--v2", "--league-diversity-manifest", str(manifest))
    assert "existing file" in capsys.readouterr().err
    manifest.write_text('{"entries": []}')
    with pytest.raises(SystemExit):
        cli("--league-diversity-manifest", str(manifest))
    assert "requires v2" in capsys.readouterr().err
    args = cli("--v2", "--league-diversity-manifest", str(manifest))
    assert args.league_diversity_manifest == Path(manifest)
