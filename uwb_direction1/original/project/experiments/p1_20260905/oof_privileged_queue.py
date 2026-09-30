"""Finite paired OOF privileged student queue with automatic auditing."""
import os
import subprocess
import sys
import time
from pipeline import BASE, RESULTS, dump, digest
from followup_analysis import load, check_pair
from oof_privileged import OUT, TAGS


def complete(arm, fold):
    tag = TAGS[arm]; result = load(tag, 2026, fold)
    if result is None:
        return False
    meta, archive = result
    baseline_meta, baseline_archive = load('sv_R_concat', 2026, fold)
    check_pair(archive, baseline_archive)
    assert meta['teacher'] is None and meta['config']['teacher'] == ''
    assert meta['initial_head_hash'] == baseline_meta['initial_head_hash']
    assert meta['parameters'] == baseline_meta['parameters']
    assert meta['config']['wrapper_sha'] == digest(BASE/'oof_privileged.py')
    assert meta['config']['proxy_sha'] == digest(OUT/f'proxy_h{fold}.npz')
    assert (RESULTS/f'{tag}_s2026_h{fold}.pt').exists()
    return True


def main():
    OUT.mkdir(exist_ok=True)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    for fold in range(1, 9):
        if not (OUT/f'proxy_h{fold}.npz').exists():
            raise FileNotFoundError(f'missing proxy fold {fold}')
        if load('sv_R_label_global', 2026, fold) is None:
            raise RuntimeError('matched global smoothing control must finish before OOF queue')
    jobs = [(arm, fold) for fold in range(1, 9) for arm in ('single', 'multi')]
    finished = {(arm, fold) for arm, fold in jobs if complete(arm, fold)}
    todo = [job for job in jobs if job not in finished]; active = {}
    try:
        while todo or active:
            while todo and len(active) < 2:
                arm, fold = todo.pop(0)
                log = (OUT/f'shape_{arm}_fold{fold}.log').open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-u', str(BASE/'oof_privileged.py'),
                                            '--folds', str(fold), '--arms', arm], cwd=BASE,
                                           stdout=log, stderr=subprocess.STDOUT)
                active[(arm, fold)] = process, log
                print('START_OOF_STUDENT', arm, fold, process.pid, flush=True)
            changed = False
            for job, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close(); del active[job]
                if code != 0 or not complete(*job):
                    raise RuntimeError(f'{job} failed rc={code}')
                finished.add(job); changed = True
                print('DONE_OOF_STUDENT', *job, flush=True)
            if changed:
                subprocess.run([sys.executable, str(BASE/'oof_privileged_summary.py')],
                               cwd=BASE, check=True)
            dump(OUT/'queue_state.json',
                 {'status': 'complete' if len(finished) == len(jobs) else 'running',
                  'queue_pid': os.getpid(), 'updated': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  'completed': [{'arm': arm, 'fold': fold} for arm, fold in sorted(finished)],
                  'expected_runs': len(jobs),
                  'active': [{'arm': arm, 'fold': fold, 'pid': process.pid}
                             for (arm, fold), (process, log) in active.items()]})
            if active:
                time.sleep(3)
        print('OOF_PRIVILEGED_ALL_COMPLETE', flush=True)
    except BaseException as exc:
        for process, log in active.values():
            if process.poll() is None:
                process.terminate()
            process.wait(); log.close()
        dump(OUT/'queue_state.json', {'status': 'failed', 'error': repr(exc),
                                      'completed': [{'arm': arm, 'fold': fold}
                                                    for arm, fold in sorted(finished)]})
        raise


if __name__ == '__main__':
    main()
