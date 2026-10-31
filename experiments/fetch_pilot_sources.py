"""Explicit download of the frozen public complementary dataset; no training."""
import json
from huggingface_hub import hf_hub_download
from experiments.deberta_lora_smoke import ROOT,digest

def main():
    cfg=json.loads((ROOT/'configs/lora_pilot200.json').read_text())
    for relative,expected in cfg['source_files_sha256'].items():
        relative=relative.replace('\\','/');name=relative.removeprefix('data/adversarial_qa/')
        path=hf_hub_download('UCLNLP/adversarial_qa',name,repo_type='dataset',revision=cfg['adversarial_revision'],
          local_dir=ROOT/'data/adversarial_qa',token=False)
        assert digest(ROOT/relative)==expected
    hf_hub_download('UCLNLP/adversarial_qa','README.md',repo_type='dataset',revision=cfg['adversarial_revision'],
      local_dir=ROOT/'data/adversarial_qa',token=False)
    print('Frozen public dataset files verified; original SQuAD cache must be present separately.')

if __name__=='__main__':main()
