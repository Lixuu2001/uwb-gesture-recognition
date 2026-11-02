"""Matched seed-2027 teacher -> Right concat KD, with resumable fold workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from pipeline import BASE, RESULTS, digest, dump, split

OUT = BASE / 'followup_results'
SEED = 2027
TAGS = ('teacher_concat', 'sv_R_concat_kd')


def complete(tag, fold):
    stem = RESULTS / f'{tag}_s{SEED}_h{fold}'
    required = [stem.with_suffix('.json'), stem.with_suffix('.pt'),
                Path(str(stem) + '_predictions.npz')]
    if tag == 'teacher_concat': required.append(Path(str(stem) + '_all_logits.npz'))
    if not all(p.is_file() for p in required):
        # train.py skips a JSON unconditionally; never silently accept partial artifacts.
        if required[0].exists(): raise RuntimeError(f'Partial artifacts: {stem}')
        return False
    r = json.loads(required[0].read_text(encoding='utf-8'))
    assert r['seed'] == SEED and r['held_subject'] == fold and r['tag'] == tag
    with np.load(required[2]) as a:
        y = np.load(BASE / 'prepared/y.npy'); s = np.load(BASE / 'prepared/subj.npy')
        for key, ids in zip(('train_ids', 'val_ids', 'ids'), split(y, s, fold, SEED)):
            np.testing.assert_array_equal(a[key], ids)
        np.testing.assert_array_equal(a['y'], y[a['ids']])
        np.testing.assert_array_equal(a['pred'], a['logits'].argmax(1))
        assert abs(float((a['pred'] == a['y']).mean()) - r['accuracy']) < 1e-12
    return True


def command(tag, fold):
    args = [sys.executable, '-u', str(BASE / 'train.py'), '--tag', tag,
            '--seed', str(SEED), '--folds', str(fold), '--microbatch', '32']
    if tag == 'teacher_concat':
        args += ['--mode', 'off', '--views', '0', '1', '2', '--readout', 'concat', '--save-teacher']
    else:
        args += ['--mode', 'concat', '--views', '2', '--teacher', 'teacher_concat']
    return args


def main():
    OUT.mkdir(exist_ok=True)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    started = time.strftime('%Y-%m-%d %H:%M:%S %z')
    active = {}; finished = set(); failure = None
    try:
        while True:
            for tag in TAGS:
                for fold in range(1, 9):
                    key = (tag, fold)
                    if key not in active and key not in finished and complete(tag, fold): finished.add(key)
            for key, (proc, handle) in list(active.items()):
                rc = proc.poll()
                if rc is None: continue
                handle.close(); del active[key]
                if rc != 0 or not complete(*key):
                    raise RuntimeError(f'{key} exited {rc}; see followup log')
                finished.add(key); print('DONE', *key, flush=True)
            for fold in range(1, 9):
                if len(active) >= 2: break
                # Complete each fold's dependency chain before starting another fold.
                for tag in TAGS:
                    key = (tag, fold)
                    if key in finished: continue
                    if key in active: break
                    if tag == TAGS[1] and (TAGS[0], fold) not in finished: break
                    log = OUT / f'{tag}_s{SEED}_h{fold}.log'
                    handle = log.open('a', encoding='utf-8')
                    proc = subprocess.Popen(command(tag, fold), cwd=BASE, stdout=handle, stderr=subprocess.STDOUT)
                    active[key] = (proc, handle)
                    print('START', *key, 'pid', proc.pid, flush=True)
                    break
            state = {'started': started, 'updated': time.strftime('%Y-%m-%d %H:%M:%S %z'),
                     'status': 'complete' if len(finished) == 16 else 'running',
                     'queue_pid': os.getpid(), 'completed': [list(k) for k in sorted(finished)],
                     'active': [{'tag': k[0], 'fold': k[1], 'pid': v[0].pid} for k, v in active.items()],
                     'expected_runs': 16, 'train_source_sha': digest(BASE / 'train.py'),
                     'plan_sha': digest(BASE / 'FIRST_PART_NEXT_PLAN.md')}
            dump(OUT / 'queue_state.json', state)
            if len(finished) == 16: break
            time.sleep(3)
    except BaseException as exc:
        failure = repr(exc)
        for proc, handle in active.values():
            if proc.poll() is None: proc.terminate()
            proc.wait(); handle.close()
        dump(OUT / 'queue_state.json', {'status': 'failed', 'error': failure,
             'completed': [list(k) for k in sorted(finished)], 'expected_runs': 16})
        raise
    print('FOLLOWUP_ALL_COMPLETE', flush=True)


if __name__ == '__main__': main()
