"""Build the standalone Colab BC notebook from tested wrapper modules."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    value['id'] = hashlib.sha256((kind + source).encode()).hexdigest()[:12]
    return value


def build():
    modules = {}
    for name in ['notebook_state_bc.py', 'notebook_state_bc_setup.py', 'notebook_state_bc_launch.py']:
        raw = (ROOT/'scripts'/name).read_bytes()
        modules[name] = {'base64': base64.b64encode(raw).decode(), 'sha256': hashlib.sha256(raw).hexdigest()}
    encoded = json.dumps(modules, separators=(',', ':'))
    cells = [cell('markdown', '''
        # 0.4秒 BC 100轮续训 · Warmup/Cosine · W&B

        在 Colab 选择 **GPU（推荐 L4）**，按顺序运行下面 **6个代码单元格**。
        保持现有183M架构和训练目标，自动恢复同一数据阶段的最新完整checkpoint。

        - W&B：在线记录 train / validation loss、选牌、动作召回、梯度、FP16 scale、内存。
        - Notebook：显示轨迹座位进度、epoch、全局step、loss和ETA。父阶段256步不冒充本阶段进度。
        - 数据：只缓存当前manifest引用的压缩shards、元数据、源码和一个最新checkpoint；训练读`/content`。
        - 学习率：350次成功更新预热至峰值3e-5，随后余弦衰减到3e-6；恢复保留调度位置。
        - 保存：每128个座位及epoch结束原子保存并校验备份latest；逐局解压缓存自动回收。

        **中断了显示：只重跑代码5。** Trainer继续在后台；重复运行代码5会连接已有进程。
        新运行时从代码1开始，自动挑选原训练目录、本notebook备份和本地缓存中最新的有效恢复点。
        `TARGET_EPOCHS=100`表示这个数据阶段的总目标，含已完成的epoch。余弦调度开始后保持相同100轮终点。
    '''), cell('markdown', '''
        ## 代码1 / 6：挂载Drive和设置目标

        默认路径指向当前0.4秒BC阶段。本notebook的续训结果写入其`bc100_cosine`子目录。
        首次运行Google会要求挂载授权；W&B登录在代码3完成，密钥只输入密码框或Colab Secret。
    '''), cell('code', '''
        from pathlib import Path
        import os, sys, json, time
        from google.colab import drive

        if not Path('/content/drive/MyDrive').is_dir():
            drive.mount('/content/drive')

        SOURCE_RUN = Path('/content/drive/MyDrive/cr_rl/runs/state_global_event040_20261110')
        NOTEBOOK_RUN = SOURCE_RUN / 'bc100_cosine'
        WORK = Path('/content/crrl_bc100_notebook')
        TARGET_EPOCHS = 100  # 当前数据阶段总目标，包含已完成epoch
        WARMUP_STEPS = 350
        MIN_LEARNING_RATE = 3e-6
        FRAME_MICROBATCH = 32
        FUSED_OPTIMIZER = True
        FLOAT32_MATMUL_PRECISION = 'high'
        DEFER_METRICS = True
        CHECKPOINT_EVERY_SEATS = 128
        VALIDATION_EVERY_SEATS = 350
        WANDB_PROJECT = 'cr-rl'
        WANDB_ENTITY = None  # 留空使用登录账号默认空间；团队账号可填team slug
        WANDB_NAME = 'bc040-183m-100ep-cosine'

        if not (SOURCE_RUN / 'dataset/dataset.manifest.json').is_file():
            raise FileNotFoundError('未找到当前阶段的数据manifest。请确认挂载的是训练使用的Google账号，或修改SOURCE_RUN。')
        WORK.mkdir(parents=True, exist_ok=True)
        NOTEBOOK_RUN.mkdir(parents=True, exist_ok=True)
        print('数据来源:', SOURCE_RUN)
        print('续训备份:', NOTEBOOK_RUN)
        print('目标总epoch:', TARGET_EPOCHS)
    '''), cell('markdown', '''
        ## 代码2 / 6：检查GPU，加载notebook工具

        只补缺少的W&B、tqdm、PyYAML，沿用Colab现有Torch。CUDA不可用会在加载大文件前停止。
    '''), cell('code', '''
        import importlib.util, subprocess, base64, hashlib
        import torch

        if not torch.version.cuda or not torch.cuda.is_available():
            raise RuntimeError('当前不是可用的GPU运行时。请在Colab“更改运行时类型”选择GPU，再从代码1运行；无需重装Torch。')
        missing = [package for module, package in [
            ('wandb', 'wandb>=0.19,<1'), ('tqdm', 'tqdm>=4.66,<5'), ('yaml', 'PyYAML>=6,<7')
        ] if importlib.util.find_spec(module) is None]
        if missing:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', *missing], check=True)
        TOOLS = WORK / 'notebook_tools'
        TOOLS.mkdir(exist_ok=True)
        EMBEDDED_MODULES = __EMBEDDED__
        for name, record in EMBEDDED_MODULES.items():
            raw = base64.b64decode(record['base64'])
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise RuntimeError('notebook工具校验失败: ' + name)
            (TOOLS / name).write_bytes(raw)
        if str(TOOLS) not in sys.path:
            sys.path.insert(0, str(TOOLS))
        from notebook_state_bc_setup import copy_verified, extract_source, stage_dataset, stage_resume, build_trainer_argv, preflight
        from notebook_state_bc import monitor_process, resume_completed, configure_wandb, log_tail
        from notebook_state_bc_launch import launch_or_attach, TrainingJob, atomic_json
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3
        RAM_LIMIT_GIB = round(max(3, min(40, total_ram * .6)), 1)
        print('CUDA:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)
        print('主机内存保护:', RAM_LIMIT_GIB, 'GiB')
    '''.replace('__EMBEDDED__', encoded)), cell('markdown', '''
        ## 代码3 / 6：W&B在线登录

        推荐在Colab左侧Secrets保存`WANDB_API_KEY`并允许本notebook访问；也可在下方密码框输入。
        不要把密钥写进单元格或聊天。登录/连接失败会明确报错，不会把离线日志冒充在线记录。
        页面会打印本次W&B运行链接。重复监控和重连运行时会复用同一个run ID。
    '''), cell('code', '''
        import getpass, uuid
        import wandb
        from IPython.display import display, Markdown

        identity_path = NOTEBOOK_RUN / 'wandb_run.json'
        if identity_path.exists():
            wb_identity = json.loads(identity_path.read_text())
            if wb_identity['project'] != WANDB_PROJECT or wb_identity.get('requested_entity') != WANDB_ENTITY:
                raise ValueError('该续训目录已有不同的W&B project/entity；请沿用原设置或另选NOTEBOOK_RUN。')
        else:
            wb_identity = {'id': uuid.uuid4().hex[:12], 'project': WANDB_PROJECT,
                           'requested_entity': WANDB_ENTITY, 'name': WANDB_NAME}
            atomic_json(identity_path, wb_identity)

        def ensure_wandb():
            current = wandb.run
            if current is not None:
                if current.id != wb_identity['id']:
                    raise RuntimeError('已有另一个W&B run；请先结束它再连接本次运行。')
                return current
            if not globals().get('_bc_wandb_logged_in', False):
                key = os.environ.get('WANDB_API_KEY')
                if not key:
                    try:
                        from google.colab import userdata
                        key = userdata.get('WANDB_API_KEY')
                    except Exception:
                        key = None
                if not key:
                    key = getpass.getpass('W&B API Key（输入隐藏）: ').strip()
                if not key:
                    raise ValueError('需要W&B登录后再开训，尚未启动trainer。')
                try:
                    authenticated = wandb.login(key=key, relogin=True)
                finally:
                    del key
                if not authenticated:
                    raise RuntimeError('W&B登录未成功，尚未启动trainer。')
            run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=wb_identity['id'],
                             resume='allow', name=wb_identity['name'], mode='online',
                             dir=str(WORK), save_code=False, settings=wandb.Settings(init_timeout=90),
                             config={'decision_seconds': 0.4, 'source_run': str(SOURCE_RUN),
                                     'target_total_epochs': TARGET_EPOCHS})
            configure_wandb(run)
            globals()['_bc_wandb_logged_in'] = True
            return run

        wb_run = ensure_wandb()
        display(Markdown('W&B：[' + wb_run.url + '](' + wb_run.url + ')'))
    '''), cell('markdown', '''
        ## 代码4 / 6：最小缓存、自动恢复、启动前检查

        只复制数据manifest引用的文件，跳过audit、历史模型、原始重建资料。
        当前训练器校验完整manifest，因此所有引用shard都要保留；不会下载整棵Drive目录。
        已校验的本地缓存直接复用。候选checkpoint只预览小型元数据，选出最新有效版本后才复制一个完整文件。
        模型、Adam、loss设置和数据指纹保持原阶段；来源不匹配会停止，不自动从零初始化。
    '''), cell('code', '''
        from tqdm.auto import tqdm
        marker = WORK / '.notebook/trainer_process.json'
        if marker.exists() and TrainingJob(marker).poll() is None:
            raise RuntimeError('trainer仍在运行；请直接重跑代码5查看进度，避免重做准备。')
        archive_candidates = [NOTEBOOK_RUN / 'source.tar.gz', SOURCE_RUN / 'source_recovery_revision2.tar.gz',
                              SOURCE_RUN / 'source.tar.gz']
        archive_drive = next((path for path in archive_candidates if path.is_file()), None)
        if archive_drive is None:
            raise FileNotFoundError('Drive中未找到本阶段源码包。')
        archive_local = WORK / 'source.tar.gz'
        copy_verified(archive_drive, archive_local)
        SRC = extract_source(archive_local, WORK)
        manifest_drive = json.loads((SOURCE_RUN / 'dataset/dataset.manifest.json').read_text())
        with tqdm(total=len(manifest_drive['shards']), desc='缓存BC数据', unit='shard') as data_bar:
            def data_progress(done, total, relative):
                data_bar.update(done - data_bar.n)
            DATASET = stage_dataset(SOURCE_RUN, WORK, progress=data_progress)
        del manifest_drive
        print('选择最新完整checkpoint…')
        staged = stage_resume(SOURCE_RUN, NOTEBOOK_RUN, WORK, manifest=DATASET,
                              extra_runs=[SOURCE_RUN / 'notebook_bc'])
        CHECKPOINT = staged['checkpoint']
        TRAIN = staged['out']
        info = staged['info']
        SEATS_PER_EPOCH = info['train_seats_per_epoch']
        INITIAL_COMPLETED = resume_completed(info['progress'], SEATS_PER_EPOCH)
        if INITIAL_COMPLETED > TARGET_EPOCHS * SEATS_PER_EPOCH:
            raise ValueError('恢复点已超过目标epoch；请在代码1提高TARGET_EPOCHS，然后重跑代码4。')
        registry = SRC / 'fixtures/split_registry.json'
        if not registry.is_file():
            raise FileNotFoundError('缺少原对局划分registry，拒绝改变训练/验证分区。')
        COMMAND = build_trainer_argv(info, source_root=SRC, dataset_manifest=DATASET,
                                    checkpoint=CHECKPOINT, out=TRAIN,
                                    backup_dir=NOTEBOOK_RUN / 'training', epochs=TARGET_EPOCHS,
                                    ram_limit_gib=RAM_LIMIT_GIB, split_registry=registry,
                                    lr_schedule='cosine', warmup_steps=WARMUP_STEPS,
                                    min_learning_rate=MIN_LEARNING_RATE,
                                    frame_microbatch=FRAME_MICROBATCH, defer_metrics=DEFER_METRICS,
                                    fused_optimizer=FUSED_OPTIMIZER,
                                    float32_matmul_precision=FLOAT32_MATMUL_PRECISION,
                                    checkpoint_every_seats=CHECKPOINT_EVERY_SEATS,
                                    validation_every_seats=VALIDATION_EVERY_SEATS)
        TRAIN_ENV = {'PYTHONPATH': str(SRC / 'src'), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                     'PYTHONDONTWRITEBYTECODE': '1'}
        print('恢复来源:', staged['source_run'])
        print('恢复进度:', info['progress'], '| 本阶段完成座位:', INITIAL_COMPLETED,
              '/', TARGET_EPOCHS * SEATS_PER_EPOCH)
        if staged['rejected_candidates']:
            print('已跳过不完整/不匹配候选:', staged['rejected_candidates'])
        print('启动检查:', preflight(COMMAND, SRC, require_cuda=True, env=TRAIN_ENV))
        wb_run = ensure_wandb()
        wb_run.config.update({'dataset_fingerprint': info['dataset_fingerprint'],
                              'target_total_epochs': TARGET_EPOCHS,
                              'resume_optimizer_step': info['progress']['optimizer_steps'],
                              'resume_epoch': info['progress']['epoch'],
                              'train_seats_per_epoch': SEATS_PER_EPOCH,
                              'source_archive_sha256': hashlib.sha256(archive_local.read_bytes()).hexdigest()},
                             allow_val_change=True)
        print('准备完成；运行代码5开始或连接训练。')
    '''), cell('markdown', '''
        ## 代码5 / 6：训练 + W&B + tqdm（可重复运行）

        打开W&B链接看曲线；下方进度条按完整轨迹座位计数，显示全局step和loss。
        停止本单元格只暂停显示，trainer仍在后台；重跑本单元格可重新连接。
        训练异常退出会打印真实`training.log`尾部。修正启动问题后，从代码4重新检查、恢复再运行。
    '''), cell('code', '''
        wb_run = ensure_wandb()
        cursor = WORK / '.notebook/wandb_cursor.json'
        remote_cursor = NOTEBOOK_RUN / 'wandb_cursor.json'
        if not cursor.exists() and remote_cursor.is_file():
            copy_verified(remote_cursor, cursor)
        sync_clock = {'last': 0.0}
        def sync_small_files(force=False):
            if not force and time.monotonic() - sync_clock['last'] < 120:
                return
            if cursor.is_file():
                copy_verified(cursor, remote_cursor)
            sync_clock['last'] = time.monotonic()
        job = launch_or_attach(COMMAND, source_root=SRC, workspace=WORK,
                               log_path=TRAIN / 'training.log', env_overrides=TRAIN_ENV)
        print('Trainer PID:', job.pid, '| W&B:', wb_run.url)
        try:
            result = monitor_process(job, TRAIN / 'metrics.jsonl', TRAIN / 'training.log',
                                     run=wb_run, target_epochs=TARGET_EPOCHS,
                                     seats_per_epoch=SEATS_PER_EPOCH,
                                     initial_completed=INITIAL_COMPLETED,
                                     cursor_path=cursor, poll_seconds=2,
                                     on_poll=sync_small_files)
            try:
                sync_small_files(force=True)
            except Exception as error:
                print('小型日志游标尚未同步:', type(error).__name__, '；训练checkpoint由trainer独立备份。')
            print(result)
            if result['logging_status'] == 'accepted':
                wb_run.finish(exit_code=0)
            else:
                print('W&B仍有待发送记录，保持run打开；稍后重跑代码5补传。')
        except KeyboardInterrupt:
            print('需要继续显示时重跑代码5。')
            raise
        except Exception:
            if job.poll() is not None:
                wb_run.finish(exit_code=1)
            raise
    '''), cell('markdown', '''
        ## 代码6 / 6：查看最近验证与进程状态

        可在暂停监控后运行。展示轨迹标签验证；实战胜率仍需固定对手对战评估。
        为减少首次加载，这版BC notebook不预载旧BC或旧快照评测模型，也不安装模拟器。
    '''), cell('code', '''
        import html
        from IPython.display import HTML
        metrics_file = TRAIN / 'metrics.jsonl'
        validations = []
        if metrics_file.is_file():
            for line in metrics_file.read_text(errors='replace').splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('phase') == 'state_bc_validation':
                    validations.append(row)
        columns = [('epoch', 'Epoch'), ('optimizer_steps', 'Step'), ('loss', 'Loss'),
                   ('slot_accuracy', '选牌准确率'), ('act_recall', '动作召回')]
        table = '<table><tr>' + ''.join('<th>' + label + '</th>' for _, label in columns) + '</tr>'
        for row in validations[-10:]:
            table += '<tr>'
            for key, label in columns:
                value = row.get(key, '')
                if key in ('slot_accuracy', 'act_recall') and isinstance(value, (int, float)):
                    value = f'{value:.2%}'
                elif key == 'loss' and isinstance(value, (int, float)):
                    value = f'{value:.4f}'
                table += '<td>' + html.escape(str(value)) + '</td>'
            table += '</tr>'
        display(HTML(table + '</table>'))
        if marker.exists():
            current_job = TrainingJob(marker)
            print('Trainer状态:', '运行中' if current_job.poll() is None else '已退出，returncode=' + str(current_job.poll()))
        else:
            print('Trainer状态: 尚未启动，运行代码5开始训练。')
        print('本地checkpoint:', TRAIN / 'latest.pt')
        print('Drive checkpoint:', NOTEBOOK_RUN / 'training/latest.pt')
        print('W&B:', wb_run.url)
        if not validations:
            print('尚无完整验证结果；启动或首个验证阶段可能仍在进行。')
    ''')]
    return {'cells': cells, 'metadata': {
        'accelerator': 'GPU', 'colab': {'name': 'colab_bc040_wandb.ipynb', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
    }, 'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    target = ROOT / 'notebooks/colab_bc040_wandb.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(target)
