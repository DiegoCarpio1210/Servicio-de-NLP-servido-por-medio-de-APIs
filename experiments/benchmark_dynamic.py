"""Estimate length-grouped dynamic-padding cost on a reproducible train-only sample."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,DataCollatorWithPadding
from qa.features import training_features,trim_training_padding
from qa.training_control import training_lock

def main():
    torch.set_num_threads(4)
    root=Path('logs/quality20261029');m=json.loads((root/'manifest.json').read_text())
    ds=load_from_disk('data/squad_v2')['train'];lookup={id:i for i,id in enumerate(ds['id'])}
    chosen=np.random.default_rng(42).choice(m['train_ids'],512,replace=False)
    ds=ds.select([lookup[id] for id in chosen])
    tok=AutoTokenizer.from_pretrained('models/continued20k',local_files_only=True)
    f=trim_training_padding(training_features(ds[:],tok),tok.model_input_names)
    order=sorted(range(len(f['input_ids'])),key=lambda i:len(f['input_ids'][i]))
    model=AutoModelForQuestionAnswering.from_pretrained('models/continued20k',local_files_only=True).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5);collate=DataCollatorWithPadding(tok,pad_to_multiple_of=8)
    rows=[];torch.cuda.reset_peak_memory_stats()
    for j in range(10):
        lo=round((len(order)-16)*max(0,j-2)/7)
        batch=collate([{k:v[i] for k,v in f.items()} for i in order[lo:lo+16]])
        batch={k:v.cuda() for k,v in batch.items()}
        torch.cuda.synchronize();start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True);model(**batch).loss.backward();optimizer.step();torch.cuda.synchronize()
        if j>=2:rows.append({'padded_length':batch['input_ids'].shape[1],'seconds':time.perf_counter()-start})
    mean=sum(r['seconds'] for r in rows)/len(rows)
    estimate=mean*2534/60
    result={'device':torch.cuda.get_device_name(),'sample_examples':512,'timed_batches':rows,
      'mean_step_seconds':mean,'estimated_2534_step_minutes':estimate,
      'estimated_minutes_with_20pct_overhead':estimate*1.2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
      'limitation':'Eight length-quantile batches approximate grouping; actual batch lengths, temperature and evaluation overhead vary.',
      'weights_saved':False,'training_stop_guard_seconds':3600}
    (root/'dynamic_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    with training_lock():main()
