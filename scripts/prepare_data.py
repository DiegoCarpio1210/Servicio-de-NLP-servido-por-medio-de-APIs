"""Download and persist official splits once, or verify the existing local snapshot."""
from pathlib import Path
from datasets import load_dataset,load_from_disk
from qa.data import stats
import json
root=Path('data/squad_v2')
ds=load_from_disk(str(root)) if root.exists() else load_dataset('rajpurkar/squad_v2',cache_dir='data/hf')
if not root.exists(): ds.save_to_disk(str(root))
Path('logs/dataset.json').write_text(json.dumps({k:stats(v) for k,v in ds.items()},indent=2))
print(Path('logs/dataset.json').read_text())
