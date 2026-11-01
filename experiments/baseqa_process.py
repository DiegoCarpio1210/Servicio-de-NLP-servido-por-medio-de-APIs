"""Windows breakaway launcher; explicit controls, no scheduled reboot restart."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from experiments.baseqa_common import ROOT,OUT,read,atomic

def launch(command,folder):
    assert os.name=='nt','Windows launcher required'
    folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.pop('QA_API_TOKEN',None);env.pop('MODEL_DIR',None);env['PYTHONUTF8']='1'
    flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
    with (folder/'stdout.log').open('ab') as out,(folder/'stderr.log').open('ab') as err:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags,close_fds=True)
    import psutil
    atomic(folder/'pid.json',{'pid':proc.pid,'create_time':psutil.Process(proc.pid).create_time(),'mechanism':'DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB'})
    print('Started independent PID',proc.pid)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','probe-child','start','stop','status']);p.add_argument('--resume',action='store_true');args=p.parse_args();os.chdir(ROOT)
    if args.action=='probe-child':
        for n in range(8):atomic(OUT/'persistence/heartbeat.json',{'pid':os.getpid(),'tick':n,'time':time.time(),'completed':n==7});time.sleep(2)
        return
    if args.action=='probe':
        launch([sys.executable,'-m','experiments.baseqa_process','probe-child'],OUT/'persistence');return
    folder=OUT/'train'
    if args.action=='status':
        print((folder/'progress.json').read_text(encoding='utf-8') if (folder/'progress.json').exists() else 'Not started');return
    if args.action=='stop':
        folder.mkdir(parents=True,exist_ok=True);(folder/'STOP').touch();print('Stop requested; finish current optimizer update and checkpoint.');return
    import psutil,socket
    if (folder/'pid.json').exists():
        record=read(folder/'pid.json')
        try:
            if abs(psutil.Process(record['pid']).create_time()-record['create_time'])<.01:raise RuntimeError('Training process already running')
        except psutil.NoSuchProcess:pass
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',8000))==0:raise RuntimeError('API on port 8000 is active; stop it yourself or postpone training')
    assert read(OUT/'persistence/verified.json')['passed'],'Persistence test must pass first'
    assert read(OUT/'smoke/mechanics.json')['passed']
    assert read(OUT/'pilot/progress.json')['state']=='completed'
    assert (ROOT/'configs/deberta_base_budget.json').exists(),'Measured budget decision required'
    if args.resume:
        assert (folder/'latest.json').exists(),'No checkpoint to resume'
        (folder/'STOP').unlink(missing_ok=True)
    elif (folder/'latest.json').exists():raise RuntimeError('Use --resume; checkpoint exists')
    launch([sys.executable,'-m','experiments.train_baseqa','--mode','train']+(['--resume'] if args.resume else []),folder)
if __name__=='__main__':main()
