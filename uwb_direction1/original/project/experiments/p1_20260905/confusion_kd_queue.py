"""Finite paired projection experiment, two workers and automatic summary."""
import json
import os
import subprocess
import sys
import time
from pipeline import BASE, RESULTS, dump, digest
from followup_analysis import load

OUT = BASE / 'confusion_kd_results'


def complete(mode, fold):
    tag = 'sv_R_project_' + mode
    result = load(tag, 2026, fold)
    if result is None: return False
    r, p = result
    assert (RESULTS / f'{tag}_s2026_h{fold}.pt').exists()
    assert r['config']['projection'] == mode
    assert r['config']['wrapper_sha'] == digest(BASE / 'confusion_kd.py')
    assert r['config']['projection_matrix_sha'] == digest(OUT / f'matrix_h{fold}.npz')
    return True


def main():
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    OUT.mkdir(exist_ok=True)
    jobs = [(m, h) for h in range(1, 9) for m in ('structure', 'uniform')]
    finished = {j for j in jobs if complete(*j)}
    todo = [j for j in jobs if j not in finished]; active = {}
    try:
        for h in range(1, 9):
            m = json.loads((OUT / f'matrix_h{h}.json').read_text())
            assert m['warnings'] == 0 and not m['validation_used_for_projection']
        while todo or active:
            while todo and len(active) < 2:
                mode, fold = todo.pop(0)
                log = (OUT / f'{mode}_h{fold}.log').open('a', encoding='utf-8')
                p = subprocess.Popen([sys.executable, '-u', str(BASE / 'confusion_kd.py'),
                        '--mode', mode, '--folds', str(fold)], cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
                active[mode, fold] = p, log
                print('START', mode, fold, 'pid', p.pid, flush=True)
            changed = False
            for key, (p, log) in list(active.items()):
                rc = p.poll()
                if rc is None: continue
                log.close(); del active[key]
                if rc != 0 or not complete(*key): raise RuntimeError(f'{key} failed rc={rc}')
                finished.add(key); changed = True; print('DONE', *key, flush=True)
            if changed:
                subprocess.run([sys.executable, str(BASE / 'confusion_kd_summary.py')], cwd=BASE, check=True)
            dump(OUT / 'queue_state.json', {'status': 'complete' if len(finished) == 16 else 'running',
                 'updated': time.strftime('%Y-%m-%d %H:%M:%S %z'), 'queue_pid': os.getpid(),
                 'completed': [list(k) for k in sorted(finished)], 'expected_runs': 16,
                 'active': [{'mode': k[0], 'fold': k[1], 'pid': p.pid} for k, (p, log) in active.items()]})
            if active: time.sleep(3)
        print('CONFUSION_ALL_COMPLETE', flush=True)
    except BaseException as exc:
        for p, log in active.values():
            if p.poll() is None: p.terminate()
            p.wait(); log.close()
        dump(OUT / 'queue_state.json', {'status': 'failed', 'error': repr(exc), 'completed': [list(k) for k in sorted(finished)]})
        raise


if __name__ == '__main__': main()
