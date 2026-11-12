"""Coordinator lifecycle and configuration, without runtime/GPU/W&B network."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/run_colab_bc100.py'
    spec=importlib.util.spec_from_file_location('run_bc100_tests',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def model_info(scheduled=False, step=606, epoch=1, next_episode=0):
    training={'chunk_size':32,'freeze_frame':False,'precision':'fp16','learning_rate':3e-5,'weight_decay':.01,
              'position_coef':1.,'tile_label_sigma':1.,'timing_objective':'natural','timing_target_action_prior':.2,
              'seed':20261110,'decision_ticks':8,'max_grad_norm':1.}
    if scheduled:
        training['lr_schedule']={'kind':'cosine','total_epochs':100,'anchor_step':606,'warmup_steps':350,'min_lr':3e-6}
    return {'progress':{'epoch':epoch,'next_episode':next_episode,'optimizer_steps':step},
            'train_seats_per_epoch':350,'dataset_fingerprint':'same-data','training_config':training,
            'model_config':{'frame_encoder_fp32':True},
            'temporal_config':{'width':1280,'heads':20,'kv_heads':5,'pattern':['kda','gqa']}}


SETTINGS={'frame_microbatch':32,'float32_matmul_precision':'high','fused_optimizer':True,'defer_metrics':True}


class FakeRun:
    id='stable-run';entity='account';url='https://wandb.ai/account/cr-rl/runs/stable-run'
    def __init__(self):self.finished=[];self.axes=[]
    def define_metric(self,*args,**kwargs):self.axes.append((args,kwargs))
    def finish(self,**kwargs):self.finished.append(kwargs)


def test_pilot_command_uses_total100_budget_and_main_does_not_repeat_handoff(tmp_path):
    runner=module();kwargs={'source_root':tmp_path/'source','root':tmp_path/'work','drive':tmp_path/'drive','settings':SETTINGS}
    pilot=runner.trainer_command(model_info(),pilot_updates=4,**kwargs)
    assert pilot[pilot.index('--epochs')+1]=='100'
    assert pilot[pilot.index('--max-episodes')+1]=='4' and '--start-lr-schedule' in pilot
    assert pilot[pilot.index('--checkpoint-every-episodes')+1]=='128'
    assert pilot[pilot.index('--validation-every-episodes')+1]=='350'
    assert '--fused-optimizer' in pilot and '--defer-metrics' in pilot
    main=runner.trainer_command(model_info(True,610,next_episode=4),**kwargs)
    assert '--start-lr-schedule' not in main and '--max-episodes' not in main
    assert runner.pilot_remaining(model_info(True,608,next_episode=2))==2
    runner.verify_pilot(model_info(),model_info(True,610,next_episode=4),4)
    with pytest.raises(RuntimeError,match='every requested successful'):
        runner.verify_pilot(model_info(),model_info(True,609,next_episode=4),4)


def test_wandb_offline_record_is_stable_and_notebook_compatible(tmp_path,monkeypatch):
    runner=module();root=tmp_path/'work';drive=tmp_path/'drive';root.mkdir();drive.mkdir()
    monkeypatch.delenv('WANDB_API_KEY',raising=False);monkeypatch.delenv('WANDB_ENTITY',raising=False)
    monkeypatch.delenv('REQUESTED_ENTITY',raising=False)
    calls=[]
    sdk=SimpleNamespace(util=SimpleNamespace(generate_id=lambda:'stable-run'),Settings=lambda **kwargs:kwargs,
                        init=lambda **kwargs:(calls.append(kwargs) or FakeRun()),
                        login=lambda **kwargs:pytest.fail('offline must not authenticate'))
    _,first=runner.open_wandb(root,drive,SETTINGS,sdk)
    _,second=runner.open_wandb(root,drive,SETTINGS,sdk)
    assert first['id']==second['id']=='stable-run'
    assert first['name']=='state_event040_bc100_cosine' and first['project']=='cr-rl'
    assert first['requested_entity'] is None and first['entity']=='account'
    assert not first['online_logging'] and first['mode']=='offline' and first['url'] is None
    assert calls[0]['mode']=='offline' and 'resume' not in calls[0]


def test_supplied_key_authentication_failure_stops_before_any_run_or_trainer(tmp_path,monkeypatch):
    runner=module();monkeypatch.setenv('WANDB_API_KEY','unit-placeholder-not-a-secret')
    sdk=SimpleNamespace(util=SimpleNamespace(generate_id=lambda:'stable-run'),login=lambda **kwargs:False,
                        init=lambda **kwargs:pytest.fail('must fail before init'))
    with pytest.raises(RuntimeError,match='authentication failed'):
        runner.open_wandb(tmp_path,tmp_path/'drive',SETTINGS,sdk)


@pytest.mark.parametrize('failure',[False,True])
def test_owned_lifecycle_runs_pilot_once_then_main_or_reports_real_failure(tmp_path,monkeypatch,failure):
    runner=module();fake_lock=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=4,flock=lambda *args:None)
    monkeypatch.setitem(sys.modules,'fcntl',fake_lock)
    snapshots=[model_info(),model_info(True,610,next_episode=4),model_info(True,35256,epoch=100)]
    current={'index':0};calls=[];run=FakeRun()
    monkeypatch.setattr(runner,'inspect_checkpoint',lambda *args,**kwargs:deepcopy(snapshots[current['index']]))
    monkeypatch.setattr(runner,'open_wandb',lambda *args:(run,{'mode':'offline'}))
    monkeypatch.setattr(runner,'preflight',lambda *args,**kwargs:{'checked':True})
    class Process:
        def __init__(self,argv,**kwargs):self.argv=argv;self.pid=99;self.done=False;calls.append(argv)
        def poll(self):return 0 if self.done else None
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    def monitor(process,*args,**kwargs):
        if failure:
            process.done=True
            raise RuntimeError('BC trainer exited with code 1. actual training traceback')
        assert kwargs['target_epochs']==100 and kwargs['cursor_path'].name=='wandb_cursor_offline.json'
        current['index']+=1;process.done=True
        return {'logging_status':'accepted'}
    monkeypatch.setattr(runner,'monitor_process',monitor)
    root,drive=tmp_path/'work',tmp_path/'drive'
    if failure:
        with pytest.raises(RuntimeError,match='actual training traceback'):
            runner.coordinate(root,drive,SETTINGS,source_root=tmp_path/'source')
        status=json.loads((drive/'status.json').read_text())
        assert status['phase']=='failed' and 'actual training traceback' in status['error']
        assert len(calls)==1 and run.finished==[{'exit_code':1}]
    else:
        result=runner.coordinate(root,drive,SETTINGS,source_root=tmp_path/'source')
        assert result['progress']['epoch']==100
        assert len(calls)==2 and '--max-episodes' in calls[0] and '--max-episodes' not in calls[1]
        assert '--start-lr-schedule' in calls[0] and '--start-lr-schedule' not in calls[1]
        status=json.loads((drive/'status.json').read_text())
        assert status['phase']=='complete' and not status['online_logging'] and status['child_pid'] is None
        assert run.finished==[{}]
