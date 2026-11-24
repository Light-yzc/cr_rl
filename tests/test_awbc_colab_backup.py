import importlib.util
from pathlib import Path


def test_backup_is_verified_and_replaces_only_complete_destination(tmp_path):
    spec=importlib.util.spec_from_file_location('awbc_backup',Path(__file__).parents[1]/'scripts/run_live_awbc_colab.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source.pt';target=tmp_path/'drive/latest.pt'
    source.write_bytes(b'valid checkpoint')
    assert module.verified_copy(source,target)
    assert target.read_bytes()==source.read_bytes()
    assert target.with_suffix('.pt.sha256').is_file()
    source.write_bytes(b'next complete checkpoint')
    assert module.verified_copy(source,target)
    assert target.read_bytes()==source.read_bytes()
