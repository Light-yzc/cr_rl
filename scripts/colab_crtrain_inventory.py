"""Read only the CR-RL runtime/data paths; no file checksums or GPU allocation."""
def main():
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import time

    def info(path):
        path = Path(path)
        return dict(path=str(path), exists=path.exists(),
                    bytes=path.stat().st_size if path.is_file() else None)

    drive = Path('/content/drive/MyDrive/cr_rl_ppo')
    report = dict(observed_unix=time.time(), cwd=os.getcwd(),
                  content_entries=[p.name for p in Path('/content').iterdir()],
                  local_free_gib=round(shutil.disk_usage('/content').free/1024**3, 2),
                  drive_mounted=drive.is_dir())
    try:
        import psutil
        report['host_available_gib'] = round(psutil.virtual_memory().available/1024**3, 2)
    except ImportError:
        pass
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                          '--format=csv,noheader'], capture_output=True, text=True)
    report['gpu'] = gpu.stdout.strip()
    paths = [drive/'ppo_latest.pt', drive/'preparation/crrl_training_code.zip',
             Path('/content/crrl_training'), Path('/content/crrl_training/prepared/input/ppo_latest.pt')]
    report['files'] = [info(p) for p in paths]
    report['preparation_entries'] = [p.name for p in (drive/'preparation').iterdir()] if (drive/'preparation').is_dir() else []
    for label, path in [('quality', drive/'preparation/full_quality_bc/state.json'),
                        ('catalog', drive/'preparation/full_quality_bc/catalog_summary.json')]:
        if path.is_file():
            data = json.loads(path.read_text())
            report[label] = {k:data.get(k) for k in ['stage','training_started','training_ready',
                'total_source_shards','completed_source_shards','accepted_games_before_cross_shard_dedup',
                'unique_games','partition_counts','split_registry_version']}
            if 'completed' in data:
                report[label]['completed_entries'] = len(data['completed'])
    report['training_processes'] = subprocess.run(
        ['ps','-eo','pid,etime,args'], capture_output=True, text=True).stdout.splitlines()
    report['training_processes'] = [s for s in report['training_processes']
        if any(x in s for x in ['train_selfplay.py', 'train_state_bc.py', 'launch_prepared_pipeline.py'])][:10]
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
