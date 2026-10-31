"""Bounded GPU feasibility only: never saves weights, trains a dataset or activates a model."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/lora-deps'))

def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def protected_hashes():
    hashes={}
    for file in ('deberta_deployment.json','quality40k_deployment.json'):
        config=json.loads((ROOT/'configs'/file).read_text())
        for name,expected in config['hashes'].items():
            path=ROOT/config['model_dir']/name;actual=digest(path)
            assert actual==expected, 'Protected file hash mismatch: '+str(path)
            hashes[str(path.relative_to(ROOT))]=actual
    for file in ('configs/active_model.json','configs/deberta_frozen_policy.json'):
        hashes[file]=digest(ROOT/file)
    return hashes

def frozen_digest(model):
    result=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            result.update(name.encode());result.update(p.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--length',type=int,choices=[64,128,256],default=128)
    parser.add_argument('--steps',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--accumulation',type=int,choices=[1,2,4,8],default=1)
    args=parser.parse_args();os.chdir(ROOT)
    branch=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'branch','--show-current'],text=True).strip()
    if branch!='experiment/deberta-lora':raise RuntimeError('Run only on experiment/deberta-lora')
    out=ROOT/'logs/deberta_lora'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
    before=protected_hashes();write(out/'protected_before.json',before)
    report={'completed':False,'config':vars(args),'precision':'FP32','seed':20261030,'branch':branch,'steps':[]}
    model=optimizer=batch=output=loss=None
    import torch,transformers,peft,accelerate
    from transformers import AutoModelForQuestionAnswering,AutoTokenizer
    from peft import LoraConfig,TaskType,get_peft_model
    from qa.features import training_features
    env={'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
      'cuda_available':torch.cuda.is_available(),'transformers':transformers.__version__,
      'peft':peft.__version__,'accelerate':accelerate.__version__,
      'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],text=True).strip(),
      'datasets_available':[str(p.relative_to(ROOT)) for p in (ROOT/'data').iterdir() if p.is_dir()]}
    write(out/'environment.json',env)
    try:
        assert torch.cuda.is_available(),'CUDA required; no silent CPU fallback'
        torch.manual_seed(20261030);torch.cuda.manual_seed_all(20261030);torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        env.update(gpu=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
          free_total_bytes=list(torch.cuda.mem_get_info()),bf16_supported_including_emulation=torch.cuda.is_bf16_supported(),
          bf16_native_supported=torch.cuda.is_bf16_supported(including_emulation=False))
        write(out/'environment.json',env);torch.cuda.reset_peak_memory_stats()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            folder=ROOT/'models/pretrained/deberta-v3-base-squad2'
            tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True)
            base=AutoModelForQuestionAnswering.from_pretrained(folder,local_files_only=True)
            report['base_parameters']=sum(p.numel() for p in base.parameters())
            modules=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and n.split('.')[-1] in ('query_proj','value_proj')]
            assert len(modules)==24 and isinstance(base.qa_outputs,torch.nn.Linear)
            report['target_modules']=modules
            config=LoraConfig(task_type=TaskType.QUESTION_ANS,r=4,lora_alpha=8,lora_dropout=.05,
              target_modules=['query_proj','value_proj'],modules_to_save=['qa_outputs'],bias='none')
            model=get_peft_model(base,config);del base
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.train()
            trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
            assert trainable and all('lora_' in n or 'qa_outputs.modules_to_save.default' in n for n in trainable)
            report.update(total_parameters=sum(p.numel() for p in model.parameters()),
              trainable_parameters=sum(p.numel() for p in trainable.values()),
              trainable_names=list(trainable),gradient_checkpointing=model.is_gradient_checkpointing)
            report['trainable_percentage']=100*report['trainable_parameters']/report['total_parameters']
            frozen_before=frozen_digest(model)
            initial={n:p.detach().clone() for n,p in trainable.items()}
            model.to('cuda')
            # Synthetic mechanics fixture, never public/500/1024 evaluation data.
            context='A calibration device in the laboratory stores its reference sample in a cobalt container.'
            answer='cobalt'
            features=training_features({'question':['What material is the container made of?'],'context':[context],
              'answers':[{'text':[answer],'answer_start':[context.index(answer)]}]},tokenizer,args.length,16)
            assert len(features['input_ids'])==1 and features['start_positions'][0]>0
            batch={k:torch.tensor(v,device='cuda') for k,v in features.items()}
            report['fixture']={'type':'synthetic mechanics only','answer':answer,'start_token':features['start_positions'][0],
              'end_token':features['end_positions'][0],'sequence_length':len(features['input_ids'][0])}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.)
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.perf_counter();losses=[]
                for _ in range(args.accumulation):
                    output=model(**batch)
                    assert output.start_logits.shape==output.end_logits.shape==(1,args.length)
                    assert torch.isfinite(output.start_logits).all() and torch.isfinite(output.end_logits).all()
                    loss=output.loss/args.accumulation;assert torch.isfinite(loss)
                    losses.append(float(output.loss.detach()));loss.backward()
                gradients={n:float(p.grad.detach().norm()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                # Default zero B initialization gives zero A gradients on the first step.
                assert any(v>0 for n,v in gradients.items() if 'lora_' in n)
                assert any(v>0 for n,v in gradients.items() if 'qa_outputs.modules_to_save' in n)
                optimizer.step();torch.cuda.synchronize()
                row={'step':step+1,'loss':sum(losses)/len(losses),'seconds':time.perf_counter()-started,
                  'nonzero_lora_gradients':sum(v>0 for n,v in gradients.items() if 'lora_' in n),
                  'gradient_norms':gradients,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                report['steps'].append(row);write(out/'results.json',report)
                with (out/'execution.log').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({k:v for k,v in row.items() if k!='gradient_norms'})+'\n')
                print(json.dumps({k:v for k,v in row.items() if k!='gradient_norms'}),flush=True)
            report['changed_trainable_names']=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(),initial[n])]
            assert any('lora_' in n for n in report['changed_trainable_names'])
            assert any('qa_outputs.modules_to_save' in n for n in report['changed_trainable_names'])
            report['frozen_parameters_unchanged']=frozen_before==frozen_digest(model)
            assert report['frozen_parameters_unchanged']
            report['warnings']=[str(w.message) for w in caught]
            report['completed']=True
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc)
        (out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        optimizer=output=loss=batch=model=None
        # References to trainable GPU parameters must also be released.
        if 'trainable' in locals():trainable.clear()
        gc.collect();torch.cuda.empty_cache()
        report['protected_files_unchanged']=protected_hashes()==before
        report['cuda_allocated_after_cleanup']=torch.cuda.memory_allocated()
        write(out/'results.json',report)
        print('Artifacts: '+str(out),flush=True)
    assert report['protected_files_unchanged']

if __name__=='__main__':
    try:main()
    except Exception:
        errors=ROOT/'logs/deberta_lora';errors.mkdir(exist_ok=True,parents=True)
        with (errors/'startup_and_execution_errors.log').open('a',encoding='utf-8') as stream:
            stream.write(time.strftime('%Y-%m-%d %H:%M:%S')+'\n'+traceback.format_exc()+'\n')
        raise
