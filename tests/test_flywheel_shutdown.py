"""Regression: monitoring must never stop collection or shut down this PC."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('flywheel_monitor',
    Path(__file__).resolve().parents[1]/'scripts/monitor_awbc_flywheel.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)

class MonitorNeverShutdownTests(unittest.TestCase):
    def check_report_only(self, failure=False):
        report = dict(unix=1, paused=False, local={'completed_games':3,'free_commit_gib':0.2},
                      cloud=None, issues=['Emulator ADB disconnected'], status='attention')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(monitor.sys,'argv',['monitor','--root',temp]), \
                 patch.object(monitor,'probe',side_effect=OSError('unavailable') if failure else None,return_value=report), \
                 patch.object(monitor,'run_bounded') as command, patch('builtins.print'):
                monitor.main()
            command.assert_not_called()
            self.assertFalse((root/'STOP').exists())
            self.assertFalse((root/'health/shutdown_pending.json').exists())
            self.assertEqual(json.loads((root/'health/latest.json').read_text())['status'],'attention')
    def test_disconnection_and_low_memory_only_report(self):
        self.check_report_only()
    def test_probe_exception_only_reports(self):
        self.check_report_only(failure=True)
    def test_shutdown_mode_is_rejected(self):
        with patch.object(monitor.sys,'argv',['monitor','--shutdown-on-failure']), patch.object(monitor.sys,'stderr'):
            with self.assertRaises(SystemExit) as error:monitor.main()
            self.assertEqual(error.exception.code,2)

if __name__=='__main__':unittest.main()
