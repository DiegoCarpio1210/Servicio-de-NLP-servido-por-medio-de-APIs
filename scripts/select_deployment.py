"""Explicit reversible selection; stop local services before switching."""
import argparse
import hashlib
import json
import socket
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['deberta','quality40k']);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    for port in (8000,8765):
        with socket.socket() as sock:
            try:sock.bind(('127.0.0.1',port))
            except OSError:raise RuntimeError('Stop the local services before switching models')
    source=root/f'configs/{args.model}_deployment.json';config=json.loads(source.read_text())
    for name,digest in config['hashes'].items():
        with (root/config['model_dir']/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=digest:raise ValueError('Model hash mismatch: '+name)
    target=root/'configs/active_model.json';temp=target.with_suffix('.tmp');temp.write_bytes(source.read_bytes());temp.replace(target)
    print('Selected '+args.model+'. Restart INICIAR_PIMIENTERO.bat.')

if __name__=='__main__':main()
