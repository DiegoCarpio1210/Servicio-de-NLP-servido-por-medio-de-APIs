"""At most 200 LoRA updates; explicit resume, atomic checkpoints, no activation."""
import argparse,gc,json,os,random,signal,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
from experiments.deberta_lora_smoke import ROOT,digest,protected_hashes
from qa.training_control import training_lock

OUT=ROOT/'logs/deberta_lora/pilot200'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def atomic(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');temp.replace(p)

def order_index(cursor,size,seed):
    epoch,offset=divmod(cursor,size);order=list(range(size));random.Random(seed+epoch).shuffle(order)
    return order[offset]

def run(resume):
    import torch,psutil
    from transformers import AutoTokenizer,AutoModelForQuestionAnswering,get_linear_schedule_with_warmup
    from peft import LoraConfig,TaskType,get_peft_model,PeftModel
    from qa.features import training_features,encode
    from qa.postprocessing import candidate
    from qa.spans import decide
    from qa.metrics import summarize,prior_weighted_metrics
    import subprocess
    branch=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'branch','--show-current'],text=True).strip()
    assert branch=='experiment/deberta-lora'
    cfg=read(OUT/'config.json');manifest_hash=digest(OUT/'manifest.json');config_hash=digest(OUT/'config.json')
    assert cfg['max_updates']==200 and cfg['accumulation']==8
    protocol=read(ROOT/'configs/lora_pilot200.json')
    assert manifest_hash==protocol['manifest_sha256']
    assert all(protocol[k]==v for k,v in cfg.items()),'Configuration changed after protocol lock'
    before=protected_hashes();assert before==read(OUT/'protected_before.json')
    data=read(OUT/'data.json');data_hash=digest(OUT/'data.json')
    assert data_hash==protocol['data_sha256'],'Prepared data changed after protocol lock'
    latest=OUT/'latest.json'
    if (OUT/'progress.json').exists() and read(OUT/'progress.json').get('state')=='completed':
        print('Pilot already completed; no further updates permitted.');return
    if latest.exists() and not resume:raise RuntimeError('Checkpoint exists; use --resume explicitly')
    if resume and not latest.exists():raise RuntimeError('No recoverable checkpoint; refusing silent restart')
    assert torch.cuda.is_available()
    torch.set_num_threads(2);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    folder=ROOT/'models/pretrained/deberta-v3-base-squad2'
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True)
    base=AutoModelForQuestionAnswering.from_pretrained(folder,local_files_only=True)
    checkpoint=OUT/read(latest)['directory'] if resume else None
    if checkpoint:
        state=torch.load(checkpoint/'trainer.pt',map_location='cpu',weights_only=False) # trusted local checkpoint only
        assert state['manifest_hash']==manifest_hash and state['config_hash']==config_hash and state['data_hash']==data_hash
        model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=True)
    else:
        model=get_peft_model(base,LoraConfig(task_type=TaskType.QUESTION_ANS,r=cfg['rank'],lora_alpha=cfg['alpha'],
          lora_dropout=cfg['dropout'],target_modules=cfg['targets'],modules_to_save=cfg['modules_to_save'],bias='none'))
        state={'step':0,'cursor':0,'history':[],'validations':[],'training_seconds':0.,'elapsed_seconds':0.,'best_step':0,'best_weighted_em':-1.}
    del base
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable)==148994
    assert all('lora_' in n or 'qa_outputs.modules_to_save.default' in n for n,p in model.named_parameters() if p.requires_grad)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.to('cuda')
    features=[]
    for row in data['train']:
        f=training_features({k:[row[k]] for k in ('question','context','answers')},tokenizer,cfg['sequence_length'],cfg['stride'])
        features.extend({k:v[i] for k,v in f.items()} for i in range(len(f['input_ids'])))
    atomic(OUT/'feature_stats.json',{'examples':len(data['train']),'features':len(features),
      'positive_features':sum(r['start_positions']!=0 for r in features),'null_features':sum(r['start_positions']==0 for r in features)})
    optimizer=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,cfg['warmup_updates'],cfg['max_updates'])
    if resume:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();old_elapsed=state['elapsed_seconds'];stopping=False
    def request_stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,request_stop)
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,request_stop)
    def log(message):
        with (OUT/'execution.log').open('a',encoding='utf-8') as stream:stream.write(message+'\n')
        print(message,flush=True)
    def save():
        name=f"checkpoint-{state['step']:04d}";target=OUT/name
        if target.exists():
            meta=read(target/'metadata.json')
            assert meta['manifest_sha256']==manifest_hash and meta['config_sha256']==config_hash
            atomic(latest,{'directory':name,'step':state['step']});return
        temp=OUT/(name+'.partial-'+uuid.uuid4().hex);temp.mkdir(exist_ok=False)
        model.save_pretrained(temp/'adapter',safe_serialization=True)
        state.update(manifest_hash=manifest_hash,config_hash=config_hash,data_hash=data_hash,
          optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
          torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),elapsed_seconds=old_elapsed+time.perf_counter()-started)
        torch.save(state,temp/'trainer.pt')
        atomic(temp/'metadata.json',{'step':state['step'],'cursor':state['cursor'],'manifest_sha256':manifest_hash,'config_sha256':config_hash,
          'adapter_sha256':digest(temp/'adapter/adapter_model.safetensors'),'training_seconds':state['training_seconds']})
        temp.replace(target);atomic(latest,{'directory':name,'step':state['step']})
    def evaluate():
        model.eval();rows=[];losses=[]
        with torch.inference_mode():
            for ex in data['development']:
                enc=encode(tokenizer,[ex['question']],[ex['context']],384,128)
                labels=training_features({k:[ex[k]] for k in ('question','context','answers')},tokenizer,384,128)
                windows=[]
                for i in range(len(enc['input_ids'])):
                    batch={k:torch.tensor([enc[k][i]],device='cuda') for k in tokenizer.model_input_names}
                    result=model(**batch,start_positions=torch.tensor([labels['start_positions'][i]],device='cuda'),end_positions=torch.tensor([labels['end_positions'][i]],device='cuda'))
                    if not torch.isfinite(result.loss):raise FloatingPointError('Nonfinite validation loss')
                    losses.append(float(result.loss));offsets=[o if s==1 else None for o,s in zip(enc['offset_mapping'][i],enc.sequence_ids(i))]
                    windows.append((result.start_logits[0].cpu().numpy(),result.end_logits[0].cpu().numpy(),offsets,enc['input_ids'][i].index(tokenizer.cls_token_id)))
                c=candidate(ex['context'],windows,cfg['policy']);response=decide(c,cfg['threshold'])
                if response['answer'] is not None:assert ex['context'][response['start']:response['start']+len(response['answer'])]==response['answer']
                rows.append({'id':ex['id'],'references':ex['answers']['text'],**c,'response':response})
        metrics=summarize(rows,cfg['threshold']);weighted=prior_weighted_metrics(metrics)
        item={'step':state['step'],'groups':metrics,'weighted_75_25':weighted,'validation_loss':float(np.mean(losses))}
        state['validations'].append(item)
        if weighted['EM']>state['best_weighted_em']:state['best_weighted_em']=weighted['EM'];state['best_step']=state['step']
        atomic(OUT/f"validation-{state['step']:04d}.json",{'metrics':item,'predictions':rows})
        atomic(OUT/'validation_metrics.json',state['validations']);log('Validation '+json.dumps(item));model.train()
    try:
        if state['step']==0 and not state['validations']:evaluate();save()
        model.train()
        while state['step']<cfg['max_updates']:
            if stopping or (OUT/'STOP').exists():save();break
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter();values=[]
            for _ in range(cfg['accumulation']):
                ix=order_index(state['cursor'],len(features),cfg['seed'])
                batch={k:torch.tensor([v],device='cuda') for k,v in features[ix].items()}
                result=model(**batch);loss=result.loss/cfg['accumulation']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                values.append(float(result.loss.detach()));loss.backward();state['cursor']+=1
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):raise FloatingPointError('Missing/nonfinite gradient')
            if not any(p.grad.abs().sum()>0 for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n):raise FloatingPointError('Zero LoRA gradients')
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            grad=float(torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True))
            optimizer.step();scheduler.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick
            state['step']+=1;state['training_seconds']+=seconds
            memory=psutil.Process().memory_info()
            state['history'].append({'step':state['step'],'loss':float(np.mean(values)),'gradient_norm':grad,'seconds':seconds,
              'learning_rate':scheduler.get_last_lr()[0],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'rss_bytes':memory.rss,'peak_rss_bytes':getattr(memory,'peak_wset',memory.rss)})
            atomic(OUT/'training_history.json',state['history'])
            atomic(OUT/'progress.json',{'state':'running','step':state['step'],'max_updates':200,'training_seconds':state['training_seconds'],'latest_checkpoint':read(latest) if latest.exists() else None})
            if state['step']%10==0:log('Training '+json.dumps(state['history'][-1]))
            if state['step'] in cfg['validation_updates']:evaluate()
            if state['step']%cfg['checkpoint_every']==0:save()
        save()
        status='completed' if state['step']==200 else 'stopped'
        atomic(OUT/'progress.json',{'state':status,'step':state['step'],'max_updates':200,'training_seconds':state['training_seconds'],
          'elapsed_seconds':old_elapsed+time.perf_counter()-started,'best_step':state['best_step'],'best_weighted_em':state['best_weighted_em'],'latest_checkpoint':read(latest)})
    except Exception:
        with (OUT/'errors.log').open('a',encoding='utf-8') as stream:stream.write(traceback.format_exc()+'\n')
        atomic(OUT/'progress.json',{'state':'failed','step':state['step'],'latest_recoverable':read(latest) if latest.exists() else None})
        raise
    finally:
        assert protected_hashes()==before
        atomic(OUT/'integrity_after.json',{'protected_files_unchanged':True,'main_expected':'02a9d34'})
        optimizer=model=result=loss=batch=None;trainable.clear();gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    with training_lock():run(args.resume)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True,parents=True)
        with (OUT/'errors.log').open('a',encoding='utf-8') as stream:stream.write(traceback.format_exc()+'\n')
        raise
