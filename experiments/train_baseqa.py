"""Bounded, recoverable LoRA QA training from Microsoft's true base encoder."""
import argparse,gc,hashlib,json,os,random,shutil,signal,time,traceback,uuid
from pathlib import Path
import numpy as np
from experiments.baseqa_common import ROOT,OUT,BASE,REVISION,read,sha,atomic,seed_all,create_model,frozen_digest
from qa.training_control import training_lock
from qa.features import encode,training_features
from qa.postprocessing import candidate
from qa.spans import decide
from qa.metrics import summarize,prior_weighted_metrics,weighted_threshold_sweep

def evaluate(model,tokenizer,examples,device='cuda'):
    import torch,psutil
    model.eval();rows=[];losses=[]
    with torch.inference_mode():
        for ex in examples:
            tick=time.perf_counter();enc=encode(tokenizer,[ex['question']],[ex['context']],384,128)
            labels=training_features({k:[ex[k]] for k in ('question','context','answers')},tokenizer,384,128);windows=[]
            for i in range(len(enc['input_ids'])):
                batch={k:torch.tensor([enc[k][i]],device=device) for k in tokenizer.model_input_names}
                result=model(**batch,start_positions=torch.tensor([labels['start_positions'][i]],device=device),end_positions=torch.tensor([labels['end_positions'][i]],device=device))
                assert torch.isfinite(result.loss);losses.append(float(result.loss))
                offsets=[o if s==1 else None for o,s in zip(enc['offset_mapping'][i],enc.sequence_ids(i))]
                windows.append((result.start_logits[0].cpu().numpy(),result.end_logits[0].cpu().numpy(),offsets,enc['input_ids'][i].index(tokenizer.cls_token_id)))
            c=candidate(ex['context'],windows,{'max_length':30,'length_penalty':0.})
            if c['text']:assert ex['context'][c['start']:c['start']+len(c['text'])]==c['text']
            rows.append({'id':ex['id'],'references':ex['answers']['text'],**c,'seconds':time.perf_counter()-tick})
    return rows,float(np.mean(losses)),psutil.Process().memory_info().rss

def checkpoint_valid(path):
    meta=read(path/'metadata.json')
    assert all(sha(path/p)==v for p,v in meta['files'].items()),'Checkpoint hash mismatch'
    return meta

def finish_evaluation(folder,cfg,data):
    """One calibration fit, then one final comparison; never writes active deployment."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForQuestionAnswering,AutoTokenizer
    from qa.deployment import load_engine
    if (folder/'final_summary.json').exists():return
    state=read(folder/'progress.json');best=folder/f"checkpoint-{state['best_step']:06d}";checkpoint_valid(best)
    seed_all(cfg['seed']);base=AutoModelForQuestionAnswering.from_pretrained(BASE,local_files_only=True)
    model=PeftModel.from_pretrained(base,best/'adapter');tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True,use_fast=True)
    model.to('cuda')
    calibration=folder/'calibration.json'
    if not calibration.exists():
        rows,loss,_=evaluate(model,tokenizer,data['calibration'])
        sweep=weighted_threshold_sweep(rows,.75)
        selected=max(sweep,key=lambda r:(r['EM'],r['F1'],-r['threshold']))
        atomic(folder/'calibration_predictions.json',rows)
        atomic(calibration,{'threshold':selected['threshold'],'objective':'weighted_75_25_EM','policy':{'max_length':30,'length_penalty':0.},
          'selected_step':state['best_step'],'adapter_sha256':sha(best/'adapter/adapter_model.safetensors'),'calibration_loss':loss})
    cal=read(calibration);assert cal['selected_step']==state['best_step']
    # Merge only this experimental model in memory for realistic CPU inference.
    model=model.merge_and_unload().to('cpu').eval();torch.cuda.empty_cache()
    results={}
    newfile=folder/'new_final_predictions.json'
    if newfile.exists():rows=read(newfile)
    else:
        rows,_,_=evaluate(model,tokenizer,data['final'],device='cpu');atomic(newfile,rows)
    metrics=summarize(rows,cal['threshold'])
    results['student_base_lora']={'metrics':metrics,'weighted_75_25':prior_weighted_metrics(metrics),
      'threshold':cal['threshold'],'mean_seconds':np.mean([r['seconds'] for r in rows]),'max_seconds':max(r['seconds'] for r in rows),
      'abstention_rate':sum(r['margin']>cal['threshold'] for r in rows)/len(rows)}
    # Export candidate separately, never activate. Allows independent API verification later.
    export=ROOT/'models/deberta-base-student'
    if not export.exists():
        tmp=export.with_name(export.name+'.partial-'+uuid.uuid4().hex);tmp.mkdir(parents=True)
        model.save_pretrained(tmp,safe_serialization=True);tokenizer.save_pretrained(tmp)
        atomic(tmp/'training_metadata.json',{'base_model':'microsoft/deberta-v3-base','source_revision':REVISION,'optimizer_steps':state['step'],
          'selected_optimizer_steps':state['best_step'],'max_seq_length':384,'doc_stride':128,'training_method':'LoRA rank4 plus new QA head'})
        atomic(tmp/'calibration.json',{**cal,'weights_sha256':sha(tmp/'model.safetensors')});tmp.replace(export)
    del model,base;gc.collect();torch.cuda.empty_cache()
    stablefile=folder/'stable_final_predictions.json'
    if stablefile.exists():old=read(stablefile)
    else:
        os.environ.pop('MODEL_DIR',None);engine=load_engine();old=[]
        for ex in data['final']:
            tick=time.perf_counter();c=engine.candidate(ex['context'],ex['question']);old.append({'id':ex['id'],'references':ex['answers']['text'],**c,'seconds':time.perf_counter()-tick})
        atomic(stablefile,old);del engine;gc.collect()
    m=summarize(old,2.7772940397262573)
    results['stable_squad_specialized']={'metrics':m,'weighted_75_25':prior_weighted_metrics(m),'mean_seconds':np.mean([r['seconds'] for r in old]),
      'max_seconds':max(r['seconds'] for r in old),'abstention_rate':sum(r['margin']>2.7772940397262573 for r in old)/len(old)}
    atomic(folder/'final_summary.json',{'results':results,'count':len(data['final']),'semantic_judge':None,
      'limitation':'Reserved official-train contexts; stable publisher trained on SQuAD train. Not vendor-unseen comparison or Claude semantic score.'})
    from experiments.report_baseqa import report
    report(folder)

def run(args):
    import torch,psutil
    from transformers import get_linear_schedule_with_warmup
    from peft import PeftModel
    cfg=read(ROOT/'configs/deberta_base_training.json');folder=OUT/args.mode;folder.mkdir(parents=True,exist_ok=True)
    if args.mode!='train':cfg={**cfg,'max_updates':2 if args.mode=='smoke' else 40,'warmup_updates':0,'runtime_hours':.5,'validation_every':40,'checkpoint_every':2 if args.mode=='smoke' else 10}
    if (folder/'progress.json').exists() and read(folder/'progress.json').get('state') in ('completed','early_stopped','budget_reached'):
        if args.mode=='train':finish_evaluation(folder,cfg,read(OUT/'data.json'))
        print('Training already finished; no further optimizer updates.');return
    latest=folder/'latest.json'
    if latest.exists() and not args.resume:raise RuntimeError('Use --resume; refusing to overwrite completed work')
    if args.resume and not latest.exists():raise RuntimeError('No checkpoint to resume')
    cfg_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();data_hash=sha(OUT/'data.json')
    assert data_hash==read(ROOT/'configs/deberta_base_data.json')['data_sha256']
    before=read(OUT/'protected_before.json');assert all(sha(ROOT/p)==h for p,h in before.items())
    assert torch.cuda.is_available();atomic(folder/'config.json',cfg)
    data=read(OUT/'data.json');seed_all(cfg['seed']);model,tokenizer,identity=create_model(cfg['seed'])
    state={'step':0,'cursor':0,'history':[],'validations':[],'elapsed_seconds':0.,'training_seconds':0.,'best_step':0,'best_loss':1e30,'stale':0}
    if args.resume:
        checkpoint=folder/read(latest)['directory'];checkpoint_valid(checkpoint)
        state=torch.load(checkpoint/'trainer.pt',map_location='cpu',weights_only=False)
        assert state['config_hash']==cfg_hash and state['data_hash']==data_hash and state['source_revision']==REVISION
        del model
        from transformers import AutoModelForQuestionAnswering
        base=AutoModelForQuestionAnswering.from_pretrained(BASE,local_files_only=True)
        model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=True);del base
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    atomic(folder/'identity.json',identity)
    model.to('cuda');trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()} if args.mode=='smoke' else None
    frozen_before=frozen_digest(model) if args.mode=='smoke' else None
    optimizer=torch.optim.AdamW([{'params':[p for n,p in trainable.items() if 'lora_' in n],'lr':cfg['lora_lr']},
      {'params':[p for n,p in trainable.items() if 'qa_outputs.modules_to_save' in n],'lr':cfg['head_lr']}],weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,cfg['warmup_updates'],cfg['max_updates'])
    if args.resume:
        optimizer.load_state_dict(state.pop('optimizer'));scheduler.load_state_dict(state.pop('scheduler'))
        random.setstate(state.pop('python_rng'));np.random.set_state(state.pop('numpy_rng'));torch.set_rng_state(state.pop('torch_rng'));torch.cuda.set_rng_state_all(state.pop('cuda_rng'))
        assert state['cursor']==state['step']*cfg['accumulation'] and len(state['history'])==state['step'] and scheduler.last_epoch==state['step']
        atomic(folder/'resume_verification.json',{'step':state['step'],'cursor':state['cursor'],'scheduler_epoch':scheduler.last_epoch,'rng_and_optimizer_restored':True})
    examples=data['train'] if args.mode=='train' else ([r for r in data['train'] if r['answers']['text']][:32]+[r for r in data['train'] if not r['answers']['text']][:32])
    development=data['development'] if args.mode=='train' else ([r for r in data['development'] if r['answers']['text']][:16]+[r for r in data['development'] if not r['answers']['text']][:16])
    features=[]
    for ex in examples:
        f=training_features({k:[ex[k]] for k in ('question','context','answers')},tokenizer,256,96)
        features.extend({k:v[i] for k,v in f.items()} for i in range(len(f['input_ids'])))
    assert all(0<=f['start_positions']<=f['end_positions']<256 for f in features)
    assert any(f['start_positions']>0 for f in features) and any(f['start_positions']==0 for f in features)
    atomic(folder/'features.json',{'examples':len(examples),'windows':len(features),'positive':sum(f['start_positions']>0 for f in features),'null':sum(f['start_positions']==0 for f in features)})
    order_cache={}
    def feature_index(cursor):
        epoch,offset=divmod(cursor,len(features))
        if epoch not in order_cache:
            order_cache.clear();order=list(range(len(features)));random.Random(cfg['seed']+epoch).shuffle(order);order_cache[epoch]=order
        return order_cache[epoch][offset]
    started=time.monotonic();old_elapsed=state['elapsed_seconds'];stopping=False;reason='stopped'
    torch.cuda.reset_peak_memory_stats()
    def request_stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    def save():
        name=f"checkpoint-{state['step']:06d}";target=folder/name
        if target.exists():checkpoint_valid(target);atomic(latest,{'directory':name,'step':state['step']});return
        temp=folder/(name+'.partial-'+uuid.uuid4().hex);temp.mkdir()
        model.save_pretrained(temp/'adapter',safe_serialization=True)
        serial={**state,'elapsed_seconds':old_elapsed+time.monotonic()-started,'config_hash':cfg_hash,'data_hash':data_hash,'source_revision':REVISION,
          'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
          'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        torch.save(serial,temp/'trainer.pt')
        files={p.relative_to(temp).as_posix():sha(p) for p in temp.rglob('*') if p.is_file()}
        atomic(temp/'metadata.json',{'step':state['step'],'data_sha256':data_hash,'config_sha256':cfg_hash,'source_revision':REVISION,'files':files})
        temp.replace(target);atomic(latest,{'directory':name,'step':state['step']})
    def progress(status):
        atomic(folder/'progress.json',{'state':status,'step':state['step'],'max_updates':cfg['max_updates'],'best_step':state['best_step'],
          'best_validation_loss':state['best_loss'],'training_seconds':state['training_seconds'],'elapsed_seconds':old_elapsed+time.monotonic()-started,
          'pid':os.getpid(),'latest':read(latest) if latest.exists() else None})
    def validate():
        rows,loss,rss=evaluate(model,tokenizer,development);m=summarize(rows,0.)
        state['validations'].append({'step':state['step'],'loss':loss,'threshold':0.,'metrics':m,'weighted_75_25':prior_weighted_metrics(m),'rss_bytes':rss})
        if loss<state['best_loss']-cfg['min_delta']:
            state['best_loss']=loss;state['best_step']=state['step'];state['stale']=0
        else:state['stale']+=1
        atomic(folder/'validation_metrics.json',state['validations']);model.train()
    try:
        if state['step']==0:
            if args.mode!='smoke':validate()
            save()
        model.train();progress('running')
        while state['step']<cfg['max_updates']:
            if stopping or (folder/'STOP').exists() or (args.stop_after and state['step']>=args.stop_after):reason='stopped';break
            if old_elapsed+time.monotonic()-started>=cfg['runtime_hours']*3600:reason='budget_reached';break
            if shutil.disk_usage(ROOT).free<6*1024**3 or psutil.virtual_memory().available<2*1024**3:reason='resource_stop';break
            if psutil.Process().memory_info().rss>12*1024**3:reason='resource_stop';break
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter();losses=[]
            for micro in range(cfg['accumulation']):
                f=features[feature_index(state['cursor'])];batch={k:torch.tensor([v],device='cuda') for k,v in f.items()}
                result=model(**batch);assert result.start_logits.shape==result.end_logits.shape==(1,256)
                if not torch.isfinite(result.loss):raise FloatingPointError('Nonfinite loss')
                losses.append(float(result.loss.detach()));(result.loss/cfg['accumulation']).backward();state['cursor']+=1
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
            assert any(p.grad.abs().sum()>0 for n,p in trainable.items() if 'lora_' in n)
            assert any(p.grad.abs().sum()>0 for n,p in trainable.items() if 'qa_outputs.modules_to_save' in n)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            grad=float(torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.,error_if_nonfinite=True))
            optimizer.step();scheduler.step();torch.cuda.synchronize();seconds=time.perf_counter()-tick
            state['step']+=1;state['training_seconds']+=seconds
            row={'step':state['step'],'loss':float(np.mean(losses)),'gradient_norm':grad,'seconds':seconds,'learning_rates':scheduler.get_last_lr(),
              'cuda_allocated_peak':torch.cuda.max_memory_allocated(),'cuda_reserved_peak':torch.cuda.max_memory_reserved(),'rss_bytes':psutil.Process().memory_info().rss}
            state['history'].append(row);atomic(folder/'training_history.json',state['history']);progress('running')
            if state['step']%10==0:
                with (folder/'execution.log').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if args.mode!='smoke' and (state['step']%cfg['validation_every']==0 or state['step']==cfg['max_updates']):validate()
            if state['step']%cfg['checkpoint_every']==0 or state['best_step']==state['step']:save()
            if state['step']>=cfg['early_stop_min_updates'] and state['stale']>=cfg['patience']:reason='early_stopped';break
        else:reason='completed'
        save();progress(reason)
        if args.mode=='smoke':
            changed=[n for n,p in trainable.items() if not torch.equal(initial[n],p.detach().cpu())]
            assert any('lora_' in n for n in changed) and any('qa_outputs.modules_to_save' in n for n in changed)
            assert frozen_digest(model)==frozen_before
            atomic(folder/'mechanics.json',{'passed':True,'frozen_backbone_unchanged':True,'QA_head_and_LoRA_changed':True,'finite_gradients':True,'valid_positive_and_null_labels':True,'trainable':148994})
        checkpoint_valid(folder/read(latest)['directory'])
    except Exception:
        with (folder/'errors.log').open('a',encoding='utf-8') as stream:stream.write(traceback.format_exc())
        progress('failed');raise
    finally:
        assert all(sha(ROOT/p)==h for p,h in before.items())
        atomic(folder/'integrity.json',{'protected_files':len(before),'unchanged':True})
    trainable.clear();optimizer=scheduler=model=result=batch=None;gc.collect();torch.cuda.empty_cache()
    from experiments.report_baseqa import report
    report(folder)
    if args.mode=='train' and reason in ('completed','early_stopped','budget_reached'):finish_evaluation(folder,cfg,data)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','pilot','train'],required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int);args=parser.parse_args()
    os.chdir(ROOT)
    with training_lock():run(args)
if __name__=='__main__':main()
