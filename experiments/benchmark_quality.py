"""Disposable warm-start copy; measure actual CUDA optimizer steps, never save its weights."""
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForQuestionAnswering,AutoTokenizer
from datasets import load_from_disk
from qa.features import training_features
from qa.training_control import training_lock


def main():
    assert torch.cuda.is_available(),'CUDA unavailable'
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    manifest=json.loads(Path('logs/quality20261029/manifest.json').read_text())
    raw=load_from_disk('data/squad_v2')['train'];lookup={id:i for i,id in enumerate(raw['id'])}
    ds=raw.select([lookup[id] for id in manifest['train_ids'][:16]])
    tok=AutoTokenizer.from_pretrained('models/continued20k',local_files_only=True)
    encoded=training_features(ds[:],tok)
    batch={k:torch.tensor(v[:16],device='cuda') for k,v in encoded.items()}
    model=AutoModelForQuestionAnswering.from_pretrained('models/continued20k',local_files_only=True).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    times=[]
    for step in range(12):
        torch.cuda.synchronize();start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True);loss=model(**batch).loss;loss.backward();optimizer.step();torch.cuda.synchronize()
        if step>=4:times.append(time.perf_counter()-start)
    mean=sum(times)/len(times)
    result={'device':torch.cuda.get_device_name(),'batch_size':16,'sequence_length':384,'measured_optimizer_steps':8,'warmup_steps':4,
      'mean_step_seconds':mean,'estimated_2534_steps_minutes':mean*2534/60,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
      'weights_saved':False,'purpose':'Timing on disposable warm-start copy; only approved training examples used; no evaluation examples'}
    Path('logs/quality20261029/gpu_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    with training_lock():main()
