"""Audit and freeze a context-disjoint complementary QA pilot before fitting."""
import hashlib,json,random,re,unicodedata
from pathlib import Path
from datasets import load_from_disk
import pyarrow.parquet as pq
from experiments.deberta_lora_smoke import ROOT,write,digest,protected_hashes

OUT=ROOT/'logs/deberta_lora/pilot200'
SEED=20261031
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).lower().split())
def ch(s):return hashlib.sha256(norm(s).encode()).hexdigest()
def qn(s):return re.sub(r'[^\w\s]','',norm(s))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Manifest already frozen; do not overwrite')
    protected=protected_hashes()
    squad=load_from_disk(str(ROOT/'data/squad_v2'))
    lookup={r['id']:r for split in squad.values() for r in split}
    old_ids=set();old_contexts=set();old_questions=set();sources=[]
    def visit(x):
        if isinstance(x,str) and x in lookup:old_ids.add(x)
        elif isinstance(x,list):
            for y in x:visit(y)
        elif isinstance(x,dict):
            if isinstance(x.get('context'),str):old_contexts.add(ch(x['context']))
            if isinstance(x.get('question'),str):old_questions.add(qn(x['question']))
            for k,v in x.items():
                if not (k.startswith('train') or k.startswith('added_train') or k=='previous_completed_model_training'):visit(v)
    paths=[p for p in (ROOT/'logs').rglob('*.json') if 'deberta_lora' not in p.parts]
    paths += [ROOT/'configs/final500_manifest.json',ROOT/'originals/public_sample.json']
    for p in paths:
        try:visit(json.loads(p.read_text(encoding='utf-8-sig')))
        except (ValueError,UnicodeError):continue
        sources.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
    for identifier in old_ids:
        old_contexts.add(ch(lookup[identifier]['context']));old_questions.add(qn(lookup[identifier]['question']))
    sources_data={};pools={};audit={'historical_ids_excluded':len(old_ids),'historical_contexts_excluded':len(old_contexts),'scanned_records':len(sources)}
    for split in ('train','validation'):
        path=ROOT/f'data/adversarial_qa/adversarialQA/{split}-00000-of-00001.parquet'
        rows=pq.read_table(path).to_pylist();sources_data[str(path.relative_to(ROOT))]=digest(path)
        valid=[];bad=duplicates=excluded=0;seen=set()
        for r in rows:
            answers=r['answers'];key=(ch(r['context']),qn(r['question']))
            if not answers['text'] or any(r['context'][a:a+len(t)]!=t for t,a in zip(answers['text'],answers['answer_start'])):
                bad+=1;continue
            if key in seen:duplicates+=1;continue
            seen.add(key)
            if r['id'] in old_ids or key[0] in old_contexts or key[1] in old_questions:excluded+=1;continue
            valid.append({**r,'source':'AdversarialQA/'+split})
        pools[split]=valid;audit['adversarial_'+split]={'total':len(rows),'invalid_or_empty_spans':bad,'duplicates':duplicates,'historical_overlap_excluded':excluded,'eligible':len(valid)}
    negatives=[{**r,'source':'SQuAD2/train'} for r in squad['train'] if not r['answers']['text'] and r['id'] not in old_ids and ch(r['context']) not in old_contexts and qn(r['question']) not in old_questions]
    audit['eligible_squad_unanswerable']=len(negatives)
    rng=random.Random(SEED)
    for rows in [*pools.values(),negatives]:rng.shuffle(rows)
    used_contexts=set();used_questions=set();partitions={}
    def take(pool,count):
        selected=[]
        for r in pool:
            context,question=ch(r['context']),qn(r['question'])
            if context in used_contexts or question in used_questions:continue
            used_contexts.add(context);used_questions.add(question);selected.append(r)
            if len(selected)==count:break
        if len(selected)!=count:raise RuntimeError(f'Insufficient valid context-disjoint data: {len(selected)}/{count}')
        return selected
    for name,count,split in [('development',64,'validation'),('calibration_reserved',64,'validation'),('train',512,'train')]:
        rows=take(pools[split],count)+take(negatives,count);rng.shuffle(rows);partitions[name]=rows
    manifest={'seed':SEED,'partitions':{k:[{'id':r['id'],'source':r['source'],'context_sha256':ch(r['context'])} for r in rows] for k,rows in partitions.items()},'audit':audit,
      'adversarial_revision':'c2d5f738db1ad21a4126a144dfbb00cb51e0a4a9','source_files_sha256':sources_data,
      'squad_fingerprints':{k:v._fingerprint for k,v in squad.items()},'license':'CC-BY-SA-4.0 for both datasets',
      'limitations':['SQuAD source was used by the vendor; AdversarialQA shares Wikipedia/SQuAD source contexts. No universal independence claim.',
        'Historical overlap audit is limited to available local records; validation is development data, not a sealed final test.',
        'Training has 50% genuine impossible questions from SQuAD2; all AdversarialQA questions have original extractive annotations.']}
    config={'seed':SEED,'max_updates':200,'learning_rate':2e-5,'warmup_updates':10,'sequence_length':256,'stride':96,
      'microbatch':1,'accumulation':8,'rank':4,'alpha':8,'dropout':.05,'precision':'FP32','gradient_checkpointing':True,
      'targets':['query_proj','value_proj'],'modules_to_save':['qa_outputs'],'validation_updates':[0,100,200],
      'checkpoint_every':25,'threshold':2.7772940397262573,'policy':{'max_length':30,'length_penalty':0.},
      'selection':'maximum development weighted75/25 EM; ties retain earlier state, including base step 0',
      'no_calibration':True,'development_count':128,'training_examples':1024,'calibration_reserved_count':128}
    write(OUT/'manifest.json',manifest);write(OUT/'config.json',config);write(OUT/'data.json',partitions)
    write(ROOT/'configs/lora_pilot200_manifest.json',manifest)
    write(OUT/'audit_sources.json',sources);write(OUT/'protected_before.json',protected)
    write(ROOT/'configs/lora_pilot200.json',{**config,'manifest_sha256':digest(OUT/'manifest.json'),'audit':audit,
      'adversarial_revision':manifest['adversarial_revision'],'source_files_sha256':sources_data,'data_sha256':digest(OUT/'data.json')})
    print(json.dumps(audit,indent=2))

if __name__=='__main__':prepare()
