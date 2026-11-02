"""Resumable task queue over (arm, fold, model_seed) with a persistent state file.

A crashed or killed fold leaves a failed entry behind instead of disappearing,
and a rerun picks up exactly where it stopped. Completion is decided by the
run's own status.json, never by the existence of a checkpoint file.
"""
import argparse
import json
import time
import traceback

import torch

import protocol as P
import train_student as TS
from protocol import RUNS, dump

STATE = RUNS.parent / 'configs/queue_state.json'


def plan(arms, folds, seeds):
    return [{'arm': a, 'fold': f, 'model_seed': s} for a in arms for s in seeds for f in folds]


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'tasks': {}, 'started': time.time()}


def key(task):
    return TS.run_name(task['arm'], task['model_seed'], task['fold'])


def status_of(task):
    path = RUNS / key(task) / 'status.json'
    if not path.exists():
        return 'pending'
    return json.loads(path.read_text(encoding='utf-8')).get('status', 'pending')


def run(arms, folds, seeds, retry_failed=False):
    for arm in arms:
        TS.spec_for(arm)
    tasks = plan(arms, folds, seeds)
    state = load_state()
    data = TS.load_student_data()
    P.verify_manifest()
    for task in tasks:
        name = key(task)
        current = status_of(task)
        if current == 'complete' or (current == 'failed' and not retry_failed):
            state['tasks'][name] = {'status': current, 'skipped': True}
            dump(STATE, state)
            print('SKIP', name, current, flush=True)
            continue
        state['tasks'][name] = {'status': 'running', 'started': time.time()}
        dump(STATE, state)
        try:
            result = TS.run(task['arm'], task['fold'], task['model_seed'], data)
            state['tasks'][name] = {'status': 'complete', 'accuracy': result['accuracy'],
                                    'macro_f1': result['macro_f1'],
                                    'elapsed_seconds': result['elapsed_seconds']}
        except Exception:
            state['tasks'][name] = {'status': 'failed', 'traceback': traceback.format_exc()[-2000:]}
            print('FAILED', name, flush=True)
            traceback.print_exc()
        dump(STATE, state)
        torch.cuda.empty_cache()
    done = sum(1 for v in state['tasks'].values() if v['status'] == 'complete')
    print('QUEUE_DONE', done, '/', len(tasks), flush=True)
    return state


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=list(P.FOLDS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[2026])
    ap.add_argument('--retry-failed', action='store_true')
    args = ap.parse_args()
    import os
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    run(args.arms, args.folds, args.seeds, args.retry_failed)
