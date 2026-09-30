"""Wait for the existing frozen queue, then audit and archive; never train.

The watchdog exits if its observed training process disappears before a
complete queue status. It cannot restart training or change the protocol.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_restored import digest,dump


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','data','prepared','benchmark','historical','audit','summary','archive','state'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    args.state.parent.mkdir(parents=True,exist_ok=True)
    lock=args.state.with_suffix('.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    tools=Path(__file__).parent
    identity=json.loads((args.runs/'identity.json').read_text())
    assert digest(tools/'circular_signal.py')==identity['script_sha256']
    protocol=tools.parent/'research/CIRCULAR_SIGNAL_PROTOCOL.json'
    assert digest(protocol)==identity['protocol_sha256']
    assert json.loads(protocol.read_text())==json.loads((args.runs/'protocol.json').read_text())
    def status(state,**extra):
        dump(args.state,dict(state=state,pid=os.getpid(),observed_unix=time.time(),**extra))
    try:
        started=time.monotonic()
        status('waiting_for_existing_training',runs=str(args.runs))
        while True:
            # queue status is small; retry a transient non-atomic JSON read.
            try:q=json.loads((args.runs/'queue_status.json').read_text())
            except json.JSONDecodeError:
                time.sleep(1);continue
            if q['state']=='complete':break
            if q['state']=='failed':raise RuntimeError('Training queue failed: '+str(q))
            proc=Path('/proc')/str(q['pid'])/'cmdline'
            if not proc.exists() or b'circular_signal.py' not in proc.read_bytes():
                raise RuntimeError('Recorded training process missing before completion')
            if time.monotonic()-started>8*3600:raise TimeoutError('Finalizer waiting limit exceeded')
            time.sleep(15)
        assert q['runs']==24
        assert digest(tools/'circular_signal.py')==identity['script_sha256']
        assert digest(protocol)==identity['protocol_sha256']
        assert json.loads(protocol.read_text())==json.loads((args.runs/'protocol.json').read_text())
        status('auditing_completed_training')
        if not args.audit.exists():
            subprocess.run([sys.executable,str(tools/'audit_circular_signal.py'),'--data',str(args.data),
                '--prepared',str(args.prepared),'--runs',str(args.runs),'--benchmark',str(args.benchmark),
                '--out',str(args.audit)],check=True)
        audit=json.loads(args.audit.read_text())
        assert audit['state']=='verified' and audit['completed_runs_audited']==24 and audit['full_experiment_complete']
        assert audit['prepared_sha256']==identity['prepared_sha256']
        status('summarizing_and_archiving')
        if not args.summary.exists():
            subprocess.run([sys.executable,str(tools/'summarize_circular_signal.py'),'--runs',str(args.runs),
                '--audit',str(args.audit),'--historical',str(args.historical),'--out',str(args.summary),
                '--archive',str(args.archive)],check=True)
        # A partial prior directory is preserved and fails explicitly.
        receipt=json.loads((args.summary/'ARCHIVE_RECEIPT.json').read_text())
        summary=json.loads((args.summary/'SUMMARY.json').read_text())
        assert summary['state']=='complete_and_audited'
        assert summary['protocol_sha256']==identity['protocol_sha256']
        assert receipt['all_archived_training_artifact_hashes_verified']
        assert digest(args.archive)==receipt['sha256']
        status('complete',runs=24,audit=str(args.audit),summary=str(args.summary),archive=str(args.archive),
               archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'],
               continuation_gate_passed=summary['frozen_continuation_gate_passed'])
    except BaseException as e:
        status('failed',error_type=type(e).__name__,error=str(e));raise


if __name__=='__main__':main()
