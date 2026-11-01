"""Isolated base-encoder experiment utilities; no production configuration writes."""
import hashlib,json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/lora-deps'))
OUT=ROOT/'logs/deberta_base_training'
BASE=ROOT/'models/base/microsoft-deberta-v3-base'
REVISION='8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(p)
def seed_all(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
def order_index(cursor,size,seed):
    epoch,offset=divmod(cursor,size);order=list(range(size));random.Random(seed+epoch).shuffle(order);return order[offset]
def verify_source():
    manifest=read(ROOT/'configs/deberta_base_source.json')
    assert manifest['repo_id']=='microsoft/deberta-v3-base' and manifest['revision']==REVISION
    for name,h in manifest['sha256'].items():assert sha(BASE/name)==h,name
    return manifest
def create_model(seed):
    import torch
    from transformers import AutoModelForQuestionAnswering,AutoTokenizer
    from peft import LoraConfig,TaskType,get_peft_model
    verify_source();seed_all(seed)
    base,loading=AutoModelForQuestionAnswering.from_pretrained(BASE,local_files_only=True,output_loading_info=True)
    missing=loading['missing_keys']
    assert 'qa_outputs.weight' in missing and 'qa_outputs.bias' in missing,missing
    initial_head=hashlib.sha256(base.qa_outputs.weight.detach().numpy().tobytes()).hexdigest()
    model=get_peft_model(base,LoraConfig(task_type=TaskType.QUESTION_ANS,r=4,lora_alpha=8,lora_dropout=.05,
        target_modules=['query_proj','value_proj'],modules_to_save=['qa_outputs'],bias='none'))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert all('lora_' in n or 'qa_outputs.modules_to_save.default' in n for n in trainable)
    assert sum(p.numel() for p in trainable.values())==148994
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    tokenizer=AutoTokenizer.from_pretrained(BASE,use_fast=True,local_files_only=True)
    return model,tokenizer,{'missing_keys':missing,'random_head_sha256':initial_head,'trainable_parameters':148994,
        'lora_parameters':147456,'qa_head_parameters':1538,'backbone_frozen':True,'source_revision':REVISION}
def frozen_digest(model):
    digest=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad:digest.update(n.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
