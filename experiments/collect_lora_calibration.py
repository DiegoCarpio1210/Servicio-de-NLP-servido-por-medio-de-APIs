"""Bounded frozen inference on the reserved calibration set. No fitting or saves of weights."""
import json,sys,time,gc,subprocess
from pathlib import Path
from experiments.deberta_lora_smoke import ROOT,digest,protected_hashes,write
from experiments.prepare_lora_pilot import ch,qn

PILOT=ROOT/'logs/deberta_lora/pilot200'
OUT=ROOT/'logs/deberta_lora/calibration_analysis'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def snapshot():
    result=protected_hashes()
    for p in PILOT.rglob('*'):
        if p.is_file() and (any(part.startswith('checkpoint-') for part in p.parts) or p.suffix=='.json'):
            result[str(p.relative_to(ROOT))]=digest(p)
    return result

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForQuestionAnswering
    from safetensors.torch import load_file
    from peft import PeftModel
    from qa.features import encode
    from qa.postprocessing import candidate
    from qa.spans import decide
    branch=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'branch','--show-current'],text=True).strip()
    assert branch=='experiment/deberta-lora'
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completed.json').exists():print('Frozen calibration inference already completed; skipping.');return
    before=snapshot();write(OUT/'integrity_before.json',before)
    data=read(PILOT/'data.json');manifest=read(PILOT/'manifest.json');protocol=read(ROOT/'experiments/lora_calibration_protocol.json')
    assert read(PILOT/'progress.json')['state']=='completed'
    assert len(data['calibration_reserved'])==128
    all_rows=[r for rows in data.values() for r in rows]
    assert len({r['id'] for r in all_rows})==len(all_rows)
    assert len({ch(r['context']) for r in all_rows})==len(all_rows)
    assert len({qn(r['question']) for r in all_rows})==len(all_rows)
    for name,rows in data.items():assert [r['id'] for r in rows]==[r['id'] for r in manifest['partitions'][name]]
    dev={r['id']:r for r in data['development']};saved={s:read(PILOT/f'validation-{s:04d}.json')['predictions'] for s in (0,100,200)}
    for rows in saved.values():
        assert [r['id'] for r in rows]==list(dev)
        for r in rows:
            ex=dev[r['id']]
            assert ex['context'][r['start']:r['start']+len(r['text'])]==r['text']
            assert decide(r,protocol['original_threshold'])==r['response']
    write(OUT/'partition_audit.json',{'train':1024,'development':128,'calibration':128,'calibration_answerable':sum(bool(r['answers']['text']) for r in data['calibration_reserved']),
      'id_context_question_overlap':0,'previous_validation_ids_match_development_only':True,'historical_exclusion_audit':manifest['audit'],
      'protocol_sha256':digest(ROOT/'experiments/lora_calibration_protocol.json')})
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed'])
    folder=ROOT/'models/pretrained/deberta-v3-base-squad2'
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True)
    model=AutoModelForQuestionAnswering.from_pretrained(folder,local_files_only=True).to('cuda').eval()
    model.requires_grad_(False);torch.cuda.reset_peak_memory_stats()
    original_head={n:p.detach().cpu().clone() for n,p in model.qa_outputs.named_parameters()}
    zero=load_file(PILOT/'checkpoint-0000/adapter/adapter_model.safetensors')
    assert all(torch.equal(zero['base_model.model.qa_outputs.'+n],p) for n,p in original_head.items())
    assert all(torch.count_nonzero(v)==0 for k,v in zero.items() if 'lora_B' in k)
    loading={'step0':{'qa_head_equals_original':True,'all_lora_B_zero':True}}
    def infer(ex):
        enc=encode(tokenizer,[ex['question']],[ex['context']],384,128);windows=[]
        with torch.inference_mode():
            for i in range(len(enc['input_ids'])):
                batch={k:torch.tensor([enc[k][i]],device='cuda') for k in tokenizer.model_input_names}
                result=model(**batch)
                offsets=[o if s==1 else None for o,s in zip(enc['offset_mapping'][i],enc.sequence_ids(i))]
                windows.append((result.start_logits[0].cpu().numpy(),result.end_logits[0].cpu().numpy(),offsets,enc['input_ids'][i].index(tokenizer.cls_token_id)))
        c=candidate(ex['context'],windows,protocol['policy'])
        assert ex['context'][c['start']:c['start']+len(c['text'])]==c['text']
        return {'id':ex['id'],'references':ex['answers']['text'],**c}
    indices=[next(i for i,r in enumerate(saved[0]) if r['references'] and r['response']['answer'] is None),
      next(i for i,r in enumerate(saved[0]) if r['references'] and r['response']['answer'] is not None),
      next(i for i,r in enumerate(saved[0]) if not r['references'])]
    def verify_predictions(step):
        errors=[]
        for i in indices:
            row=saved[step][i];actual=infer(dev[row['id']]);error=abs(actual['margin']-row['margin'])
            assert actual['text']==row['text'] and actual['start']==row['start'] and error<2e-4
            errors.append(error)
        return {'checked_development_examples':len(indices),'maximum_margin_difference':max(errors)}
    def collect(name):
        target=OUT/(name+'_calibration.json')
        if target.exists():
            rows=read(target);assert [r['id'] for r in rows]==[r['id'] for r in data['calibration_reserved']];return
        rows=[];start=time.perf_counter()
        for i,ex in enumerate(data['calibration_reserved']):
            rows.append(infer(ex))
            if (i+1)%32==0:print(f'{name}: {i+1}/128',flush=True)
        write(target,rows);loading.setdefault(name,{})['calibration_seconds']=time.perf_counter()-start
    loading['original']=verify_predictions(0);collect('original')
    for step in (100,200):
        name='step'+str(step);path=PILOT/f'checkpoint-{step:04d}/adapter'
        metadata=read(path.parent/'metadata.json');assert digest(path/'adapter_model.safetensors')==metadata['adapter_sha256']
        if step==100:model=PeftModel.from_pretrained(model,path,adapter_name=name,is_trainable=False)
        else:model.load_adapter(path,adapter_name=name,is_trainable=False)
        model.set_adapter(name);model.eval();model.requires_grad_(False)
        tensors=load_file(path/'adapter_model.safetensors');runtime=dict(model.named_parameters());verified=0
        for key,value in tensors.items():
            target=key.replace('.lora_A.weight',f'.lora_A.{name}.weight').replace('.lora_B.weight',f'.lora_B.{name}.weight')
            if '.qa_outputs.' in target:target=target.replace('.qa_outputs.',f'.qa_outputs.modules_to_save.{name}.')
            assert torch.equal(runtime[target].detach().cpu(),value),target
            verified+=1
        head_diff=sum(float((tensors['base_model.model.qa_outputs.'+k]-v).square().sum()) for k,v in original_head.items())**.5
        assert head_diff>0 and any(torch.count_nonzero(v)>0 for k,v in tensors.items() if 'lora_B' in k)
        loading[name]={'all_loaded_tensors_verified':verified,'active_adapter':model.active_adapter,'qa_head_l2_change':head_diff,
          'adapter_sha256':metadata['adapter_sha256'],**verify_predictions(step)}
        if step==200:collect(name)
    write(OUT/'loading_verification.json',loading)
    assert snapshot()==before
    write(OUT/'completed.json',{'completed':True,'protected_files_unchanged':True,'calibration_examples_per_model':128,
      'gradient_updates':0,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
    model=None;gc.collect();torch.cuda.empty_cache();print('Frozen calibration inference and checkpoint loading checks completed.',flush=True)

if __name__=='__main__':main()
