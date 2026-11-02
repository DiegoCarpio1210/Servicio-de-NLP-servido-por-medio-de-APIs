"""Pin models/deberta-full-fast as the active deployment and archive its training evidence in logs/.

Run from the repository root after experiments.train_fast has finished:
    python -m scripts.activate_full_fast
Weights stay out of Git (models/ is ignored); only hashes and evidence are written.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "models/deberta-full-fast"
BASE = "microsoft/deberta-v3-base"
REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
FILES = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "spm.model"]
EVIDENCE = ["training_history.json", "loss.png", "results.json", "training_metadata.json", "calibration.json"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    model = ROOT / MODEL
    meta = json.loads((model / "training_metadata.json").read_text(encoding="utf-8"))
    cal = json.loads((model / "calibration.json").read_text(encoding="utf-8"))
    hashes = {name: sha256(model / name) for name in FILES}
    if cal["weights_sha256"] != hashes["model.safetensors"]:
        raise SystemExit("calibration.json does not belong to these weights")
    if meta["base_model"] != BASE or meta["source_revision"] != REVISION:
        raise SystemExit("The model was not trained from the pinned microsoft/deberta-v3-base checkpoint")

    target = ROOT / "configs/active_model.json"
    previous = json.loads(target.read_text(encoding="utf-8"))
    archive = ROOT / "configs/previous_active_model_student3500.json"
    if not archive.exists() and previous.get("model_dir") != MODEL:
        archive.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")

    keys = ("base_model", "source_revision", "optimizer_steps", "max_seq_length", "doc_stride", "training_method")
    active = {
        "model_dir": MODEL,
        "training_steps": meta["optimizer_steps"],
        "hashes": hashes,
        "metadata": {**{k: meta[k] for k in keys}, "selected_optimizer_steps": meta["optimizer_steps"]},
        "calibration": {"threshold": cal["threshold"], "objective": cal["objective"],
                        "policy": cal["policy"], "weights_sha256": cal["weights_sha256"]},
        "source": previous["source"],
        "evaluation_key": "deberta-full-fast",
    }

    logs = ROOT / "logs/deberta_full_fast"
    logs.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE:
        shutil.copy2(model / name, logs / name)
    target.write_text(json.dumps(active, indent=2) + "\n", encoding="utf-8")
    print(f"Active model: {MODEL} ({meta['optimizer_steps']} steps, threshold {cal['threshold']:.4f})")
    print(f"Evidence copied to {logs.relative_to(ROOT)}")


if __name__ == "__main__":
    main()