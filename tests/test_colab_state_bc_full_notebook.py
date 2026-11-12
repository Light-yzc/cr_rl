"""Verify the released full-data notebook without Drive, CUDA, or tensors."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location(
        "full_notebook_builder_test", ROOT / "scripts/make_colab_state_bc_full_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_cells():
    return ["".join(c["source"]) for c in builder().build_full()["cells"] if c["cell_type"] == "code"]


def assignments(source, names, namespace):
    """Execute only whitelisted top-level assignments, never mounts or training."""
    body = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    exec(compile(ast.Module(body=body, type_ignores=[]), "<notebook-settings>", "exec"), namespace)
    return namespace


def command():
    cells = code_cells()
    namespace = assignments(cells[0], {
        "SOURCE_RUN", "RELEASE", "NOTEBOOK_RUN", "WORK", "TARGET_EPOCHS", "MIN_TRAIN_GAMES",
        "EXPECTED_TOTAL_GAMES", "EPISODE_BATCH_SIZE", "FRAME_MICROBATCH", "VALIDATION_FRACTION",
        "VALIDATION_GAMES", "EARLY_STOPPING_PATIENCE", "EARLY_STOPPING_MIN_DELTA",
        "LEARNING_RATE", "MIN_LEARNING_RATE", "WARMUP_STEPS", "LR_SCHEDULE",
    }, {"Path": Path})
    namespace.update(sys=sys, SRC=ROOT, COLLECTION_ROOT=Path("/local/data"), RAM_LIMIT_GIB=8)
    assignments(cells[3], {"PARENT_STAGE", "PARENT", "PARENT_MANIFEST", "REGISTRY", "COMMAND"}, namespace)
    return namespace["COMMAND"]


def test_full_notebook_is_reproducible_and_embeds_exact_helpers():
    module = builder()
    built = module.build_full()
    assert json.loads(module.TARGET.read_text(encoding="utf-8")) == built
    assert len(code_cells()) == 6
    assert built["metadata"]["colab"]["name"] == "colab_bc040_full.ipynb"
    for cell in built["cells"]:
        assert cell["metadata"] == {}
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), "<full-notebook>", "exec")
    embedded = assignments(code_cells()[1], {"EMBEDDED_MODULES"}, {})["EMBEDDED_MODULES"]
    for name, record in embedded.items():
        raw = base64.b64decode(record["base64"])
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        assert raw == (ROOT / "scripts" / name).read_bytes()
    # Secrets stay in environment/Colab Secret/getpass, never a saved value.
    assert "getpass.getpass" in code_cells()[2]
    assert "userdata.get('WANDB_API_KEY')" in code_cells()[2]


def test_full_notebook_preserves_full_collection_and_isolates_backend_stage():
    argv = command()
    def value(flag):
        return argv[argv.index(flag) + 1]
    assert value("--expected-total-games") == "10654"
    assert value("--expected-source-shards") == "333"
    assert value("--episode-batch-size") == "8"
    assert value("--frame-microbatch") == "256"
    assert value("--validation-fraction") == "0.04"
    assert value("--validation-games") == "16"
    assert value("--epochs") == "36"
    assert '--allow-epoch-extension' in argv
    assert value("--patience") == "0"
    assert value("--learning-rate") == "0.0003"
    assert value("--min-learning-rate") == "0.0003"
    assert value("--warmup-steps") == "0"
    assert "--upgrade-placement" in argv
    assert value("--placement-cnn-channels") == "64"
    assert value("--out").endswith("experiment_placement_priorcnn64_20261111")
    assert value("--parent-checkpoint").replace("\\", "/").endswith(
        "experiment_v04_16/stage_000_be9eead6f870/training/latest.pt"
    )
    assert "initial_weights" not in " ".join(argv)
    assert "--resume" not in argv  # The controller chooses same-stage resume itself.
    for cell in code_cells()[:4]:
        assert "launch_or_attach(COMMAND" not in cell
    assert "restart_completed=waiting" in code_cells()[4]


def test_full_notebook_cuda_guard_stops_before_loading_tools(monkeypatch, tmp_path):
    torch = ModuleType("torch")
    torch.version = SimpleNamespace(cuda=None)
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="GPU"):
        exec(code_cells()[1], {"WORK": tmp_path, "os": __import__("os"), "sys": sys})
    assert not (tmp_path / "notebook_tools").exists()


def test_full_notebook_progress_uses_current_stage_absolute_seats():
    source = code_cells()[4]
    assert "completed_seats({**record,'phase':'state_bc_train'},seats_per_epoch)" in source
    assert "'state_bc_progress'" in source
    assert "bar.update(1)" not in source
    assert "full_progress.json" in source
    assert "int(progress_state.get('completed_seats',0))" in source
    assert "torch.load" not in source


def test_full_notebook_command_parses_with_shipped_controller():
    spec = importlib.util.spec_from_file_location("full_controller_notebook_test", ROOT / "scripts/run_state_bc_full.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(command()[2:])
    assert args.upgrade_placement and args.placement_cnn_channels == 64
