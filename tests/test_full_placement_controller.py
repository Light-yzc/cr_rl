"""Verify the user-facing controller actually invokes the supported training CLI."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_placement_controller_upgrade_and_resume_arguments(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    controller = module("run_state_bc_full")
    trainer = module("train_state_bc")
    args = controller.parse_args([
        "--manifest", "dataset.json", "--out", str(tmp_path / "output"),
        "--local-work", str(tmp_path / "work"), "--parent-checkpoint", "parent.pt",
        "--parent-manifest", "parent_dataset.json", "--split-registry", "registry.json",
        "--upgrade-placement", "--placement-cnn-channels", "64", "--warmup-steps", "0",
        "--learning-rate", "0.0003", "--min-learning-rate", "0.0003",
        "--episode-batch-size", "8", "--frame-microbatch", "256", "--device", "cpu",
    ])
    for resume in (False, True):
        command = controller.training_argv(args, Path("dataset.json"), tmp_path / "stage",
            Path("registry.json"), resume=resume, parent=Path("parent.pt"),
            parent_manifest=Path("parent_dataset.json"), seats=12)
        parsed = trainer.parse_args(command[2:])
        assert parsed.lr_schedule == "constant" and parsed.learning_rate == .0003
        assert parsed.episode_batch_size == 8 and parsed.frame_microbatch == 256
        assert parsed.warmup_steps is None and parsed.min_learning_rate is None
        if resume:
            assert parsed.resume == tmp_path / "stage/training/latest.pt"
            assert parsed.upgrade_placement_from is None
        else:
            assert parsed.upgrade_placement_from == Path("parent.pt")
            assert parsed.init_state_dataset_manifest == Path("parent_dataset.json")


def test_existing_joint_controller_keeps_cosine_contract(tmp_path):
    controller = module("run_state_bc_full")
    args = controller.parse_args([
        "--manifest", "dataset.json", "--out", str(tmp_path / "out"),
        "--local-work", str(tmp_path / "work"), "--parent-checkpoint", "parent.pt",
        "--parent-manifest", "parent_dataset.json", "--split-registry", "registry.json",
    ])
    command = controller.training_argv(args, Path("dataset.json"), tmp_path / "stage",
        Path("registry.json"), resume=False, parent=Path("parent.pt"),
        parent_manifest=Path("parent_dataset.json"), seats=12)
    assert command[command.index("--lr-schedule") + 1] == "cosine"
    assert command[command.index("--warmup-steps") + 1] == "350"
    assert "--init-from-state" in command and "--upgrade-placement-from" not in command


def test_existing_controller_accepts_only_explicit_unfreeze_extension(tmp_path, monkeypatch):
    controller=module('run_state_bc_full')
    original={'episode_batch_size':8,'epochs':8,'learning_rate':.0003,'seed_registry_sha256':'same'}
    schedule={'checkpoints':4,'checkpoint_interval':512,'learning_rate':1e-5}
    requested={**original,'unfreeze_schedule':schedule}
    state={'contract':original,'completed_epochs':0,'pending':{'child_pid':123}}
    controller.allow_unfreeze_contract_extension(state,requested,4)
    assert state['contract']==requested and len(state['unfreeze_changes'])==1
    assert state['pending']=={'child_pid':123}
    for changed in ({**requested,'episode_batch_size':4},
                    {**requested,'unfreeze_schedule':{**schedule,'checkpoints':3}}):
        controller.allow_unfreeze_contract_extension(state,changed,3)
        assert state['contract']==requested


def test_epoch_extension_preserves_cursor_and_only_increases_same_constant_run():
    import copy
    import pytest
    controller=module('run_state_bc_full')
    old={'epochs':8,'episode_batch_size':8,'learning_rate':.0003,
         'placement_stage':{'lr_schedule':'constant'}}
    pending={'start_epoch':5,'train_seats':20446,'child_pid':123}
    state={'contract':old,'completed_epochs':5,'pending':pending}
    new={**old,'epochs':36}
    controller.allow_epoch_contract_extension(state,new,False)
    assert state['contract']==old
    controller.allow_epoch_contract_extension(state,new,True)
    assert state['contract']==new and state['pending'] is pending
    assert state['epoch_extensions'][0]['previous_epochs']==8
    controller.allow_epoch_contract_extension(state,new,True)
    assert len(state['epoch_extensions'])==1
    for bad in ({**new,'epochs':24},{**new,'episode_batch_size':16,'epochs':40},
                {**new,'epochs':40,'placement_stage':{'lr_schedule':'cosine'}}):
        with pytest.raises(ValueError,match='only increase'):
            controller.allow_epoch_contract_extension(state,bad,True)
    assert state['contract']==new and state['pending']==pending
