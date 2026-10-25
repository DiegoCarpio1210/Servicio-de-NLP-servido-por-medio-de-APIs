"""Estimate cost from actual overflow count and measured GPU throughput, before training."""
import json,math
from pathlib import Path
from transformers import AutoTokenizer
from qa.data import load_splits,stats,validation_partitions
from qa.features import training_features
c=json.loads(Path('configs/subset12k.json').read_text())
raw=load_splits(c); parts=validation_partitions(raw['validation'],c)
tok=AutoTokenizer.from_pretrained(c['base_model'],revision=c['base_revision'],cache_dir='cache',local_files_only=True)
f=raw['train'].map(lambda x:training_features(x,tok,c['max_seq_length'],c['doc_stride']),batched=True,remove_columns=raw['train'].column_names)
benchmark=json.loads(Path('logs/subset12k/gpu_benchmark_fp32.json').read_text())
measurement=next(r for r in benchmark['rows'] if r['batch']==c['batch_size'])
steps=math.ceil(math.ceil(len(f)/c['batch_size'])/c['gradient_accumulation_steps'])*c['epochs']
seconds=steps*measurement['mean_microstep_seconds']*c['gradient_accumulation_steps']
result={'train':stats(raw['train']),'validation':{k:stats(v) for k,v in parts.items()},'training_windows':len(f),'optimizer_steps':steps,
        'conservative_training_minutes':seconds/60,'estimated_pipeline_minutes_range':[round(seconds/60),round(seconds/60+25)],
        'assumptions':'Measured static 384-token FP32 throughput. Dynamic padding may reduce time. Evaluation, checkpoints and CPU scoring add overhead.',
        'batch':c['batch_size'],'gradient_accumulation_steps':c['gradient_accumulation_steps'],'precision':'fp32','dynamic_padding':True}
Path('logs/subset12k/estimate.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
