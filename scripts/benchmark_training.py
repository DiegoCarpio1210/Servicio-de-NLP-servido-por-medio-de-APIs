"""Bounded training-only GPU throughput probe; discarded weights never become a candidate."""
import gc,json,time,argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,set_seed
from qa.data import load_splits
from qa.features import training_features

parser=argparse.ArgumentParser(); parser.add_argument('--precision',choices=['fp16','fp32'],default='fp16'); parser.add_argument('--batches',type=int,nargs='+',default=[4,8,16]); args=parser.parse_args()
c=json.loads(Path('configs/subset12k.json').read_text())
probe=c.copy(); probe.update(train_size=128,validation_size=32)
ds=load_splits(probe)['train']
tokenizer=AutoTokenizer.from_pretrained(c['base_model'],revision=c['base_revision'],cache_dir='cache',local_files_only=True)
f=training_features(ds[:],tokenizer,c['max_seq_length'],c['doc_stride'])
torch.set_num_threads(4)
assert torch.cuda.is_available(), 'CUDA is required'
results=[]
for batch in args.batches:
    set_seed(c['seed']); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model=AutoModelForQuestionAnswering.from_pretrained(c['base_model'],revision=c['base_revision'],cache_dir='cache',local_files_only=True).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=args.precision=='fp16')
    inputs={k:torch.tensor(v[:batch],device='cuda') for k,v in f.items()}
    try:
        elapsed=[]
        for step in range(5):
            optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16,enabled=args.precision=='fp16'): loss=model(**inputs).loss
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
            if step: elapsed.append(time.perf_counter()-start)
        result={'batch':batch,'mean_microstep_seconds':sum(elapsed)/len(elapsed),'examples_per_second':batch/(sum(elapsed)/len(elapsed)),
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'status':'ok'}
    except torch.OutOfMemoryError:
        result={'batch':batch,'status':'out_of_memory'}
    results.append(result); print(json.dumps(result),flush=True)
    del model,optimizer,scaler,inputs
    if 'loss' in locals(): del loss
    gc.collect(); torch.cuda.empty_cache()
root=Path('logs/subset12k'); root.mkdir(parents=True,exist_ok=True)
(root/('gpu_benchmark_'+args.precision+'.json')).write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'rows':results,'purpose':'performance probe only; weights discarded'},indent=2))
