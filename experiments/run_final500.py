"""Run the three frozen comparisons sequentially in fresh bounded subprocesses."""
import json,os,subprocess,sys,time
from pathlib import Path
from qa.training_control import training_lock
from experiments.final500 import ROOT,OUT,MODELS,atomic

def main():
    os.chdir(ROOT);OUT.mkdir(exist_ok=True)
    with training_lock(OUT/'run.lock'):
        frozen=json.loads((ROOT/'configs/final500_manifest.json').read_text(encoding='utf-8'))
        if not (OUT/'manifest.json').exists():atomic(OUT/'manifest.json',frozen)
        assert json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))==frozen, 'Manifest differs from the committed selection'
        for key in MODELS:
            folder=OUT/key;folder.mkdir(exist_ok=True)
            if (folder/'summary.json').exists():continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            atomic(OUT/'job.json',{'pid':os.getpid(),'model':key,'state':'running'})
            with (folder/'execution.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u','-m','experiments.final500',key],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
            if result.returncode:
                atomic(OUT/'job.json',{'pid':os.getpid(),'model':key,'state':'failed','returncode':result.returncode})
                raise SystemExit(result.returncode)
        atomic(OUT/'job.json',{'pid':os.getpid(),'state':'completed'})

if __name__=='__main__':main()
