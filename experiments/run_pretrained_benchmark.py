"""Resumable, bounded CPU-only benchmark. Never activates or trains a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/pretrained_benchmark'
CONFIG=ROOT/'configs/pretrained_benchmark.json'

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def event(stage,**fields):
    atomic(OUT/'progress.json',{'pid':os.getpid(),'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields})
    logging.info('%s %s',stage,fields)
def error(label):
    with (OUT/'errors.log').open('a',encoding='utf-8') as f:f.write(label+'\n'+traceback.format_exc()+'\n')
    logging.exception(label)

def partition(rows,seed,calibration_size,final_size):
    """Balanced labels, entire-context exclusion, deterministic ID-only manifest."""
    groups={}
    for r in rows:groups.setdefault(r['context'],[]).append(r)
    keys=list(groups);rng=random.Random(seed);rng.shuffle(keys);result={}
    for name,size in [('calibration',calibration_size),('final',final_size)]:
        bins={False:[],True:[]};contexts=[]
        while min(map(len,bins.values()))<size//2:
            if not keys:raise ValueError('Insufficient disjoint contexts')
            context=keys.pop();contexts.append(hashlib.sha256(context.encode()).hexdigest())
            for r in groups[context]:bins[bool(r['answers']['text'])].append(r['id'])
        ids=[]
        for flag in (False,True):rng.shuffle(bins[flag]);ids+=bins[flag][:size//2]
        rng.shuffle(ids);result[name]={'ids':ids,'context_hashes':contexts}
    return result

def threshold(rows,null_floor=None):
    from qa.metrics import weighted_threshold_sweep
    points=weighted_threshold_sweep(rows,.75)
    if null_floor is not None:
        nulls=[r['margin'] for r in rows if not r['references']]
        points=[p for p in points if sum(x>p['threshold'] for x in nulls)/len(nulls)>=null_floor]
    return max(points,key=lambda p:(p['EM'],p['F1'],-abs(p['threshold'])))['threshold']

def measure(rows,t):
    from qa.metrics import summarize,prior_weighted_metrics
    groups=summarize(rows,t)
    return {'groups':groups,'weighted_75_25':prior_weighted_metrics(groups),
      'abstention_rate':sum(r['margin']>t for r in rows)/len(rows),
      'answerability_classification_accuracy':sum((r['margin']>t)==(not r['references']) for r in rows)/len(rows),
      'abstention_accuracy_unanswerable_EM':groups['unanswerable']['EM']}

class BenchmarkEngine:
    def __init__(self,folder,policy,threads):
        import torch
        from transformers import AutoTokenizer,AutoModelForQuestionAnswering
        torch.set_num_threads(threads)
        self.tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True,trust_remote_code=False)
        if not self.tokenizer.is_fast:raise ValueError('Fast character offsets required')
        self.model=AutoModelForQuestionAnswering.from_pretrained(folder,local_files_only=True,trust_remote_code=False).cpu().eval()
        self.policy=policy;self.lock=threading.Lock();self.threshold=0.
        self.params=sum(p.numel() for p in self.model.parameters());self.max_length=384;self.stride=128
        self.metadata={'base_model':str(folder),'optimizer_steps':0};self.calibration={'objective':'weighted_75_25_EM'}
        self.weights_sha256=sha(Path(folder)/'model.safetensors')

    def candidate(self,context,question):
        import torch
        from qa.features import encode
        from qa.postprocessing import candidate
        with self.lock,torch.inference_mode():
            batch=encode(self.tokenizer,[question],[context],384,128);windows=[]
            for i in range(len(batch['input_ids'])):
                inputs={k:torch.tensor([batch[k][i]]) for k in self.tokenizer.model_input_names}
                out=self.model(**inputs)
                offsets=[o if seq==1 else None for o,seq in zip(batch['offset_mapping'][i],batch.sequence_ids(i))]
                cls=batch['input_ids'][i].index(self.tokenizer.cls_token_id)
                windows.append((out.start_logits[0].numpy(),out.end_logits[0].numpy(),offsets,cls))
        result=candidate(context,windows,self.policy)
        if result['start'] is not None:assert context[result['start']:result['start']+len(result['text'])]==result['text']
        return result

    def predict(self,context,question):
        from qa.spans import decide
        return decide(self.candidate(context,question),self.threshold)

def collect(engine,lookup,ids,path):
    rows=read(path) if path.exists() else []
    assert [r['id'] for r in rows]==ids[:len(rows)],'Resume IDs differ'
    for i in range(len(rows),len(ids)):
        ex=lookup[ids[i]]
        rows.append({'id':ex['id'],'references':ex['answers']['text'],**engine.candidate(ex['context'],ex['question'])})
        if (i+1)%16==0 or i+1==len(ids):
            atomic(path,rows);atomic(path.parent/'progress.json',{'stage':path.stem,'completed':i+1,'total':len(ids),'pid':os.getpid()})
            logging.info('%s %s/%s',path,i+1,len(ids))
    return rows

def http_benchmark(engine,out):
    import httpx
    import secrets
    import uvicorn
    from jsonschema import validate
    import yaml
    from qa.api import create_app
    from scripts.http_contract_checks import check
    with socket.socket() as probe:probe.bind(('127.0.0.1',8000))
    token=secrets.token_urlsafe(32);os.environ['QA_API_TOKEN']=token
    server=uvicorn.Server(uvicorn.Config(create_app(loader=lambda:engine),host='127.0.0.1',port=8000,log_level='error'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    try:
        with httpx.Client(base_url='http://127.0.0.1:8000',timeout=5,trust_env=False) as c:
            for _ in range(100):
                try:
                    if c.get('/health').status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
            else:raise TimeoutError('Local HTTP startup failed')
            contract_error=None
            try:check(c,token,out)
            except Exception as exc:contract_error=type(exc).__name__;error('HTTP contract '+str(out))
        items=read(ROOT/'originals/public_sample.json')['items']
        schema=yaml.safe_load((ROOT/'originals/qa_api_contract.yaml').read_text(encoding='utf-8'))['components']['schemas']['PredictResponse']
        def request(i):
            item=items[i%5];start=time.perf_counter()
            try:
                r=httpx.post('http://127.0.0.1:8000/predict',headers={'Authorization':'Bearer '+token},json={k:item[k] for k in ('id','context','question')},timeout=5,trust_env=False)
                r.raise_for_status();body=r.json();validate(body,schema)
                assert body['id']==item['id'] and (body['answer'] is None)==(body['start'] is None)
                if body['answer'] is not None:assert item['context'][body['start']:body['start']+len(body['answer'])]==body['answer']
                return {'id':item['id'],'seconds':time.perf_counter()-start,'response':body,'valid':True}
            except Exception as exc:return {'id':item['id'],'seconds':time.perf_counter()-start,'valid':False,'error':type(exc).__name__}
        results={'contract_error':contract_error,'transport':'real loopback HTTP; no tunnel','concurrency':2}
        for name,workers in [('sequential',1),('concurrent',2)]:
            start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(request,range(50)))
            total=time.perf_counter()-start
            results[name]={'rows':rows,'elapsed_seconds':total,'max_seconds':max(r['seconds'] for r in rows),
              'pass':total<=150 and all(r['valid'] and r['seconds']<=5 for r in rows)}
        return results
    finally:
        server.should_exit=True;thread.join(timeout=15);os.environ.pop('QA_API_TOKEN',None)

def worker(key):
    import psutil
    from datasets import load_from_disk
    from huggingface_hub import snapshot_download
    config=read(CONFIG);out=OUT/key;out.mkdir(exist_ok=True)
    if psutil.virtual_memory().available<config['minimum_available_ram_gib']*2**30:raise MemoryError('Insufficient available RAM')
    if shutil.disk_usage(ROOT).free<config['minimum_free_disk_gib']*2**30:raise OSError('Insufficient disk space')
    if key=='quality40k':
        active=read(ROOT/'configs/quality40k_deployment.json');folder=ROOT/active['model_dir'];policy=active['calibration']['policy']
    else:
        item=next(m for m in config['models'] if m['id'].split('/')[-1]==key)
        folder=ROOT/'models/pretrained'/key;policy={'max_length':30,'length_penalty':0.}
        marker=out/'download.json'
        if not marker.exists():
            atomic(out/'progress.json',{'stage':'downloading','revision':item['revision']})
            snapshot_download(item['id'],revision=item['revision'],local_dir=folder,token=False,max_workers=1,
              allow_patterns=['config.json','model.safetensors','tokenizer*','special_tokens_map.json','vocab.json','vocab.txt','merges.txt','spm.model','sentencepiece*','README.md','LICENSE*'])
            assert (folder/'model.safetensors').is_file()
            atomic(marker,{'revision':item['revision'],'weights_sha256':sha(folder/'model.safetensors'),'license':item['license']})
    engine=BenchmarkEngine(folder,policy,config['cpu_threads'])
    if key=='quality40k':assert engine.weights_sha256==active['hashes']['model.safetensors']
    manifest=read(OUT/'manifest.json');ds=load_from_disk(str(ROOT/'data/squad_v2'))['validation'];lookup={r['id']:r for r in ds}
    frozen=out/'policy.json'
    if not frozen.exists():
        if key=='quality40k':thresholds={'active':active['calibration']['threshold']}
        else:
            cal=collect(engine,lookup,manifest['calibration']['ids'],out/'calibration_predictions.json')
            thresholds={'weighted_EM':threshold(cal),'null95_calibration':threshold(cal,.95)}
        atomic(frozen,{'thresholds':thresholds,'policy':policy,'weights_sha256':engine.weights_sha256,'manifest_sha256':sha(OUT/'manifest.json'),'before_final':True})
    frozen_data=read(frozen);assert frozen_data['weights_sha256']==engine.weights_sha256 and frozen_data['manifest_sha256']==sha(OUT/'manifest.json')
    final=collect(engine,lookup,manifest['final']['ids'],out/'final_predictions.json')
    primary='active' if key=='quality40k' else 'weighted_EM';engine.threshold=frozen_data['thresholds'][primary]
    http_path=out/'http.json'
    if not http_path.exists():
        try:atomic(http_path,http_benchmark(engine,out))
        except Exception as exc:error(key+' HTTP');atomic(http_path,{'error':type(exc).__name__,'pass':False})
    memory=psutil.Process().memory_info()
    result={'model':key,'weights_sha256':engine.weights_sha256,'params':engine.params,'device':'cpu','primary_profile':primary,
      'profiles':{k:measure(final,t) for k,t in frozen_data['thresholds'].items()},'policy':frozen_data,
      'rss_mib':memory.rss/2**20,'peak_process_working_set_mib':getattr(memory,'peak_wset',memory.rss)/2**20,
      'http':read(http_path),'final_count':len(final),'independence_limitations':manifest['limitations'],'completed':True}
    atomic(out/'metrics.json',result)

def kill_tree(proc):
    import psutil
    try:
        parent=psutil.Process(proc.pid);owned=[*parent.children(recursive=True),parent]
        for p in reversed(owned):
            try:p.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(owned,timeout=5)
        for p in alive:p.kill()
    except psutil.NoSuchProcess:pass

def orchestrate():
    import psutil
    from datasets import load_from_disk
    config=read(CONFIG)
    for name in ('torch','transformers','datasets','sentencepiece','google.protobuf','psutil','httpx','uvicorn'):
        if importlib.util.find_spec(name) is None:raise ImportError('Missing dependency: '+name)
    OUT.mkdir(exist_ok=True);(OUT/'errors.log').touch(exist_ok=True)
    atomic(OUT/'pid.json',{'pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    frozen_config=OUT/'config.json'
    if frozen_config.exists():assert read(frozen_config)==config,'Configuration changed; use a new benchmark directory'
    else:atomic(frozen_config,config)
    manifest=OUT/'manifest.json'
    if not manifest.exists():
        rows=load_from_disk(str(ROOT/'data/squad_v2'))['validation']
        m=partition(rows,config['seed'],512,1024)
        m['dataset_fingerprint']=rows._fingerprint
        m['limitations']=['Official SQuAD2 validation only; no official-train holdout because pretrained models saw SQuAD training.',
          'All official validation contexts were consulted historically. Vendor published scores also use validation; this is not a blind official test.',
          'Calibration/final contexts disjoint within this benchmark; quality40k prior selection/calibration may overlap these historical contexts.',
          'No final-based threshold tuning. Null95 is a frozen secondary operating point, not guaranteed 95% final accuracy.',
          'Historical abandoned full-data run exposure remains uncertain. Cross-domain professor performance is unknown.']
        atomic(manifest,m)
    keys=[m['id'].split('/')[-1] for m in config['models']]+['quality40k'];states={}
    for key in keys:
        target=OUT/key;target.mkdir(exist_ok=True)
        if (target/'metrics.json').exists():states[key]='completed_cached';continue
        event('running',model=key)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_DISABLE_XET='1',HF_HUB_DOWNLOAD_TIMEOUT='60',HF_HUB_ETAG_TIMEOUT='30',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1',OMP_NUM_THREADS='2')
        with (target/'execution.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','experiments.run_pretrained_benchmark','--worker',key],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic();reason=None
            while proc.poll() is None:
                try:
                    p=psutil.Process(proc.pid);rss=sum(x.memory_info().rss for x in [p,*p.children(recursive=True)])
                    if time.monotonic()-start>config['worker_timeout_seconds']:reason='time limit'
                    elif rss>config['max_worker_rss_gib']*2**30 or psutil.virtual_memory().available<2*2**30:reason='RAM limit'
                    if reason:kill_tree(proc);break
                except psutil.NoSuchProcess:break
                time.sleep(2)
            code=proc.wait(timeout=15)
        states[key]='completed' if code==0 and (target/'metrics.json').exists() else 'failed: '+str(reason or code)
        if not states[key].startswith('completed'):
            with (OUT/'errors.log').open('a',encoding='utf-8') as f:f.write(key+': '+states[key]+'; see candidate execution.log\n')
        event('stage_finished',model=key,status=states[key])
    results={k:read(OUT/k/'metrics.json') for k in keys if (OUT/k/'metrics.json').exists()}
    atomic(OUT/'metrics.json',results)
    summary={k:{'metrics':v['profiles'][v['primary_profile']],'latency':{n:{a:b for a,b in v['http'].get(n,{}).items() if a!='rows'} for n in ('sequential','concurrent')},'ram_mib':v['peak_process_working_set_mib']} for k,v in results.items()}
    ranking=sorted(results,key=lambda k:results[k]['profiles'][results[k]['primary_profile']]['weighted_75_25']['EM'],reverse=True)
    atomic(OUT/'comparison.json',{'states':states,'same_final_ids':True,'models':summary,'ranked_by_weighted_EM':ranking,
      'recommendation':'Review highest-ranked measured model, its impossible EM and HTTP limits; no automatic replacement. Targets are not guaranteed.',
      'limitations':read(manifest)['limitations'],'production_unchanged':True})
    event('completed' if len(results)==len(keys) else 'completed_with_errors',states=states)

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(OUT/'execution.log',encoding='utf-8'),logging.StreamHandler()])
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    try:
        if args.worker:worker(args.worker)
        else:
            from qa.training_control import training_lock
            with training_lock(OUT/'benchmark.lock'):orchestrate()
    except Exception as exc:
        if not args.worker and 'already holds the project lock' in str(exc):
            logging.warning('Another benchmark owns the lock; no competing worker launched.')
            raise SystemExit(2)
        error('Fatal '+str(args.worker or 'orchestrator'))
        if not args.worker:event('failed',reason='See errors.log')
        raise SystemExit(1)

if __name__=='__main__':main()
