"""Finite eight-fold matched-global label-smoothing queue."""
import os
import subprocess
import sys
import time
from pipeline import BASE, RESULTS, dump, digest
from followup_analysis import load, check_pair
from label_global import TAG, OUT


def complete(fold):
    result = load(TAG, 2026, fold)
    if result is None:
        return False
    meta, archive = result
    adaptive_meta, adaptive_archive = load('sv_R_label_uniform', 2026, fold)
    check_pair(archive, adaptive_archive)
    assert meta['teacher'] is None
    assert meta['initial_head_hash'] == adaptive_meta['initial_head_hash']
    assert meta['parameters'] == adaptive_meta['parameters']
    assert meta['config']['wrapper_sha'] == digest(BASE/'label_global.py')
    assert (RESULTS/f'{TAG}_s2026_h{fold}.pt').exists()
    assert (OUT/f'global_matrix_h{fold}.npz').exists()
    return True


def main():
    OUT.mkdir(exist_ok=True)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    finished = {fold for fold in range(1, 9) if complete(fold)}
    todo = [fold for fold in range(1, 9) if fold not in finished]
    active = {}
    try:
        while todo or active:
            while todo and len(active) < 2:
                fold = todo.pop(0); log = (OUT/f'fold{fold}.log').open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-u', str(BASE/'label_global.py'),
                                            '--folds', str(fold)], cwd=BASE,
                                           stdout=log, stderr=subprocess.STDOUT)
                active[fold] = process, log
                print('START_GLOBAL_CONTROL', fold, process.pid, flush=True)
            changed = False
            for fold, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del active[fold]
                if code != 0 or not complete(fold):
                    raise RuntimeError(f'fold {fold} failed rc={code}')
                finished.add(fold); changed = True
                print('DONE_GLOBAL_CONTROL', fold, flush=True)
            if changed:
                subprocess.run([sys.executable, str(BASE/'label_smoothing_summary.py')],
                               cwd=BASE, check=True)
            dump(OUT/'queue_state.json',
                 {'status': 'complete' if len(finished) == 8 else 'running',
                  'queue_pid': os.getpid(), 'updated': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  'completed': sorted(finished), 'expected_runs': 8,
                  'active': [{'fold': fold, 'pid': process.pid}
                             for fold, (process, log) in active.items()]})
            if active:
                time.sleep(3)
        print('GLOBAL_CONTROL_ALL_COMPLETE', flush=True)
    except BaseException as exc:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
            process.wait(); log.close()
        dump(OUT/'queue_state.json', {'status': 'failed', 'error': repr(exc),
                                      'completed': sorted(finished)})
        raise


if __name__ == '__main__':
    main()
