"""Rebuild only the published frozen selection, never resample after results."""
import hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from datasets import load_from_disk
from experiments.prepare_lora_pilot import ROOT,OUT,ch
from experiments.deberta_lora_smoke import write,digest,protected_hashes

def frozen_bytes(value,expected):
    raw=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    for newline in ('\n','\r\n'):
        data=raw.replace('\n',newline).encode('utf-8')
        if hashlib.sha256(data).hexdigest()==expected:return data
    raise ValueError('Reconstructed content does not match the frozen digest')

def main():
    protocol=json.loads((ROOT/'configs/lora_pilot200.json').read_text())
    manifest_path=ROOT/'configs/lora_pilot200_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest_bytes=frozen_bytes(manifest,protocol['manifest_sha256'])
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data.json').exists():
        assert digest(OUT/'data.json')==protocol['data_sha256'];print('Existing frozen data verified; nothing overwritten.');return
    pools={}
    for split in ('train','validation'):
        path=ROOT/f'data/adversarial_qa/adversarialQA/{split}-00000-of-00001.parquet'
        expected=next(v for k,v in protocol['source_files_sha256'].items() if k.replace('\\','/')==path.relative_to(ROOT).as_posix())
        assert digest(path)==expected
        pools['AdversarialQA/'+split]={r['id']:r for r in pq.read_table(path).to_pylist()}
    squad=load_from_disk(str(ROOT/'data/squad_v2'))
    assert all(squad[k]._fingerprint==v for k,v in manifest['squad_fingerprints'].items())
    pools['SQuAD2/train']={r['id']:r for r in squad['train']}
    data={}
    for part,selection in manifest['partitions'].items():
        data[part]=[]
        for item in selection:
            row={**pools[item['source']][item['id']],'source':item['source']}
            assert ch(row['context'])==item['context_sha256'];data[part].append(row)
    (OUT/'data.json').write_bytes(frozen_bytes(data,protocol['data_sha256']))
    (OUT/'manifest.json').write_bytes(manifest_bytes)
    metadata={'manifest_sha256','audit','adversarial_revision','source_files_sha256','data_sha256'}
    config=json.dumps({k:v for k,v in protocol.items() if k not in metadata},indent=2)+'\n'
    (OUT/'config.json').write_bytes(config.replace('\n','\r\n' if b'\r\n' in manifest_bytes else '\n').encode())
    write(OUT/'protected_before.json',protected_hashes());print('Frozen data reconstructed and hash-verified.')

if __name__=='__main__':main()
