"""Explicit pinned download; never runs as part of application startup."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/deberta_deployment.json').read_text())
    source=config['source'];folder=root/config['model_dir']
    snapshot_download(source['repo_id'],revision=source['revision'],local_dir=folder,
                      allow_patterns=list(config['hashes']),token=False,max_workers=1)
    for name,expected in config['hashes'].items():
        with (folder/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
                raise ValueError('Downloaded file hash mismatch: '+name)
    print('Pinned DeBERTa weights and tokenizer verified. No activation or training performed.')

if __name__=='__main__':main()
