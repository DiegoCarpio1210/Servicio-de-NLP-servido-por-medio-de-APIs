"""Durable single-run launcher; saves checkpoints independently of the chat."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from qa.training_control import training_lock


def main():
    root=Path('logs/continued20k')
    job=root/'job.json'
    if job.exists(): raise RuntimeError('This authorized run was already launched; inspect job/checkpoints instead')
    status={'pid':os.getpid(),'state':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save(): job.write_text(json.dumps(status,indent=2))
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save()
        with (root/'training_execution.txt').open('w') as stream:
            proc=subprocess.Popen([sys.executable,'-m','scripts.train','--config','configs/continued20k.json'],stdout=stream,stderr=subprocess.STDOUT)
            status['child_pid']=proc.pid; save(); code=proc.wait()
        status.update(returncode=code,state='completed_pending_calibration' if code==0 else 'stopped_with_checkpoint' if code==75 else 'failed')
        status['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    with training_lock('logs/continued20k/pipeline.lock'): main()
