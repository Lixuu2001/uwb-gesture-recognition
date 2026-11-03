"""Isolated synthetic child fixtures. No activation, model, UWB or Drive access."""
from pathlib import Path
import json, os, subprocess, sys, tempfile, unittest

HERE = Path(__file__).resolve().parent

class SupervisorBoundary(unittest.TestCase):
    def fixture(self, child, *, wall=10, rss=262144, cpu=(5, 6), launch=False,
                oom=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'profile_cost.py').write_text(child)
            policy = dict(profile_CPU_soft_seconds=cpu[0],
                          profile_CPU_hard_seconds=cpu[1],
                          profile_wall_seconds=wall, RSS_stop_KiB=rss)
            wrapper = '\n'.join([
                'import sys,json',
                'from pathlib import Path',
                'sys.path.insert(0, ' + repr(str(HERE)) + ')',
                'import independent_supervisor as module',
                'module.B=Path(' + repr(directory) + ')',
                'module.ready=lambda: ' + repr(policy),
                ('module.subprocess.Popen=lambda *a,**k: '
                 '(_ for _ in ()).throw(OSError("synthetic launch failure"))')
                    if launch else '',
                ('observations=iter([{"oom_kill":0},{"oom_kill":1}]); '
                 'module.events=lambda:next(observations)') if oom else '',
                'record=module.bounded_run([sys.executable,"-u",str(module.B/"profile_cost.py")],module.B/"supervisor01",'+repr(wall)+','+repr(cpu[0])+','+repr(cpu[1])+','+repr(rss)+'); sys.exit(0 if record["exit_code"]==0 and record["resource_stop_reason"] is None else 1)', 
            ])
            completed = subprocess.run([sys.executable, '-c', wrapper],
                                       capture_output=True, text=True, timeout=15)
            path = root / 'supervisor01/TERMINATION.json'
            self.assertTrue(path.exists(), completed.stderr)
            record = json.loads(path.read_text())
            self.assertFalse(record['accuracy_measured'])
            self.assertFalse(record['scientific_negative'])
            if record['pid'] is not None:
                self.assertFalse(Path('/proc', str(record['pid'])).exists())
            return completed.returncode, record

    def test_real_success_records_exit_cpu_rss_cgroup_and_reaps(self):
        code, record = self.fixture('import time\ntime.sleep(.3)\n')
        self.assertEqual(code, 0)
        self.assertEqual(record['exit_code'], 0)
        self.assertIsNone(record['resource_stop_reason'])
        self.assertGreater(record['peak_observed_RSS_KiB'], 0)
        self.assertGreaterEqual(record['child_cpu_seconds'], 0)
        self.assertEqual(record['cgroup_oom_kill_delta'], 0)
        self.assertTrue(record['cgroup_events_available'])

    def test_launch_failure_records_evidence(self):
        code, record = self.fixture('', launch=True)
        self.assertNotEqual(code, 0)
        self.assertIsNone(record['pid'])
        self.assertEqual(record['resource_stop_reason'], 'child_launch_failed')
        self.assertEqual(record['error_type'], 'OSError')

    def test_wall_limit_terminates_only_own_child(self):
        sentinel = subprocess.Popen([sys.executable, '-c',
                                     'import time; time.sleep(10)'])
        try:
            code, record = self.fixture('import time\ntime.sleep(10)\n', wall=.15)
            self.assertNotEqual(code, 0)
            self.assertTrue(record['wall_timeout'])
            self.assertEqual(record['resource_stop_reason'], 'wall_resource_limit')
            self.assertIsNone(sentinel.poll())
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=3)

    def test_rss_limit_retains_resource_failure_without_scientific_claim(self):
        code, record = self.fixture('import time\nx=bytearray(20*1024*1024)\ntime.sleep(10)\n', rss=1024)
        self.assertNotEqual(code, 0)
        self.assertEqual(record['resource_stop_reason'], 'RSS_resource_limit')
        self.assertGreater(record['peak_observed_RSS_KiB'], 1024)

    def test_real_cpu_limit_cannot_report_success(self):
        code, record = self.fixture('while True: pass\n', cpu=(1, 2))
        self.assertNotEqual(code, 0)
        self.assertLess(record['exit_code'], 0)
        self.assertGreater(record['child_cpu_seconds'], .5)

    def test_child_exception_cannot_report_success(self):
        code, record = self.fixture('raise RuntimeError("synthetic child failure")\n')
        self.assertNotEqual(code, 0)
        self.assertEqual(record['exit_code'], 1)

    def test_virtual_cgroup_oom_delta_refuses_zero_exit(self):
        code, record = self.fixture('pass\n', oom=True)
        self.assertNotEqual(code, 0)
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(record['cgroup_oom_kill_delta'], 1)
        self.assertEqual(record['resource_stop_reason'], 'cgroup_OOM_kill_recorded')

if __name__ == '__main__':
    unittest.main()
