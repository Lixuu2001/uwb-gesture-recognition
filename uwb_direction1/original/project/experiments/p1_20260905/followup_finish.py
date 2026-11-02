"""One-shot dependency chain: await batch A, train sign controls, summarize.

This is a finite local experiment pipeline, with no messaging or recurring job.
"""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pipeline import BASE, dump
from followup_queue import OUT


def await_batch_a():
    state = json.loads((OUT / 'queue_state.json').read_text())
    if state['status'] == 'complete': return
    if state['status'] != 'running': raise RuntimeError(f'Batch A failed: {state}')
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, state['queue_pid'])
    if not handle: raise RuntimeError('Batch A process not found')
    try:
        while True:
            rc = kernel.WaitForSingleObject(handle, 15000)
            if rc == 0: break
            if rc != 258: raise RuntimeError(f'Cannot wait for batch A: {rc}')
    finally:
        kernel.CloseHandle(handle)
    state = json.loads((OUT / 'queue_state.json').read_text())
    if state['status'] != 'complete': raise RuntimeError(f'Batch A did not complete: {state}')


def state(status, **kwargs):
    dump(OUT / 'continuation_state.json', {'status': status, 'pid': os.getpid(),
         'updated': time.strftime('%Y-%m-%d %H:%M:%S %z'), **kwargs})


def summarize():
    subprocess.run([sys.executable, str(BASE / 'followup_analysis.py')], cwd=BASE, check=True)


def main():
    state('waiting_for_batch_a', expected_sign_control_folds=8)
    active = {}
    try:
        await_batch_a(); summarize()
        from parallel_core import done
        todo = [h for h in range(1, 9) if not done('sv_R_phase_abs', 2026, h)]
        completed = [h for h in range(1, 9) if h not in todo]
        while todo or active:
            while todo and len(active) < 2:
                h = todo.pop(0)
                log = (OUT / f'sv_R_phase_abs_s2026_h{h}.log').open('a', encoding='utf-8')
                proc = subprocess.Popen([sys.executable, '-u', str(BASE / 'phase_sign_control.py'),
                         '--folds', str(h)], cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
                active[h] = (proc, log); print('START_SIGN_CONTROL', h, proc.pid, flush=True)
            for h, (proc, log) in list(active.items()):
                rc = proc.poll()
                if rc is None: continue
                log.close(); del active[h]
                if rc != 0 or not done('sv_R_phase_abs', 2026, h):
                    raise RuntimeError(f'Sign control fold {h} failed rc={rc}')
                completed.append(h); summarize(); print('DONE_SIGN_CONTROL', h, flush=True)
            state('sign_control_running', completed=sorted(completed),
                  active=[{'fold': h, 'pid': proc.pid} for h, (proc, log) in active.items()])
            if active: time.sleep(3)
        summarize()
        state('complete', completed=sorted(completed), results=str(OUT / 'RESULTS.md'))
        print('FOLLOWUP_A_B_COMPLETE', flush=True)
    except BaseException as exc:
        for proc, log in active.values():
            if proc.poll() is None: proc.terminate()
            proc.wait(); log.close()
        state('failed', error=repr(exc))
        raise


if __name__ == '__main__': main()
