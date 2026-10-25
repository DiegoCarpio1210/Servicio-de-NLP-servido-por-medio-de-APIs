"""Authorized 12k/2-epoch pipeline; never expands training automatically."""
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path
from qa.training_control import training_lock


def main():
    p=argparse.ArgumentParser(); p.add_argument('--resume'); p.add_argument('--evaluate-only',action='store_true'); args=p.parse_args()
    root=Path('logs/subset12k'); root.mkdir(parents=True,exist_ok=True)
    now=datetime.datetime.now(datetime.timezone.utc)
    stamp=now.strftime('%Y%m%dT%H%M%SZ')
    status={'state':'running','started_utc':now.isoformat(),'pid':os.getpid(),'stages':[],'config':'configs/subset12k.json'}
    def save(): (root/'job.json').write_text(json.dumps(status,indent=2))
    train=[sys.executable,'-m','scripts.train','--config','configs/subset12k.json']
    if args.resume: train+=['--resume',args.resume]
    commands=[] if args.evaluate_only else [('training',train)]
    commands += [('calibration',[sys.executable,'-m','scripts.evaluate','--model-dir','models/subset12k']),
                 ('http',[sys.executable,'-m','scripts.verify_local','--model-dir','models/subset12k','--repeat','10']),
                 ('report',[sys.executable,'-m','scripts.summarize_candidate'])]
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        save()
        for name,command in commands:
            status['current_stage']=name
            logfile=root/(stamp+'_'+name+'_execution.txt'); status['current_log']=str(logfile); save()
            with logfile.open('w') as output:
                proc=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
                status['child_pid']=proc.pid; save(); code=proc.wait()
            status['stages'].append({'name':name,'returncode':code,'log':str(logfile)}); save()
            if code and name!='http':
                status['state']='stopped_with_checkpoint' if code==75 else 'failed'; save(); return code
        status['state']='completed_requires_review'; save(); return 0
    finally:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    with training_lock('logs/subset12k/pipeline.lock'): raise SystemExit(main())
