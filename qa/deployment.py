"""Pinned local deployment; never downloads or trains during API startup."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def configuration():
    return json.loads((ROOT/'configs/active_model.json').read_text(encoding='utf-8'))

def load_engine():
    from qa.inference import Engine
    active=configuration()
    selected=Path(os.getenv('MODEL_DIR',str(ROOT/active['model_dir']))).resolve()
    return Engine(selected,deployment=active if selected==(ROOT/active['model_dir']).resolve() else None)
