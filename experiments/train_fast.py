"""Fast full fine-tuning of microsoft/deberta-v3-base for SQuAD 2.0 on a 6 GB laptop GPU.

Designed for an RTX 4050 (6 GB, bf16) and a total budget of about one hour.
It produces a model folder that qa.api serves unchanged through MODEL_DIR.

Run from the repository root:
  1) Speed probe (about 3 minutes; tells you how many examples fit your budget):
       python -m experiments.train_fast --probe
  2) Real run (use the number the probe recommends):
       python -m experiments.train_fast --train-examples 60000 --compare-dir models/deberta-base-student

Decisions (all chosen for quality per minute on a 6 GB GPU):
  * Full fine-tuning instead of LoRA rank 4: the whole encoder adapts to QA.
  * Word embeddings frozen (98M of 184M params): saves ~1.2 GB of optimizer state and time,
    with negligible quality loss for an English-only task.
  * bf16 mixed precision (Ada GPUs support it; fp16 is unstable for DeBERTa-v3).
  * Dynamic padding + group_by_length: most SQuAD windows are far shorter than 384 tokens.
  * max_seq_length 384 / doc_stride 128 in training AND serving (the old model trained at 256/96).
  * No-answer threshold calibrated on validation articles disjoint from the reported ones,
    for a 75% answerable / 25% unanswerable mix (the hidden test set's announced mix).
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from qa.features import encode, training_features, trim_training_padding
from qa.metrics import score
from qa.postprocessing import candidate

ROOT = Path(__file__).resolve().parents[1]
BASE = "microsoft/deberta-v3-base"
REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LEN, STRIDE = 384, 128
POLICY = {"max_length": 30, "length_penalty": 0.0}
SQUAD_TRAIN_SIZE = 130319


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_squad(name):
    return load_from_disk(name) if Path(name).is_dir() else load_dataset(name)


def tokenize_train(dataset, tokenizer):
    names = tokenizer.model_input_names

    def features(batch):
        return trim_training_padding(training_features(batch, tokenizer, MAX_LEN, STRIDE), names)

    return dataset.map(features, batched=True, remove_columns=dataset.column_names,
                       desc="Tokenizing training windows")


class Clock(TrainerCallback):
    """Stops at the time budget and measures seconds per optimizer step."""

    def __init__(self, minutes, skip=10):
        self.limit, self.skip, self.t0, self.mark, self.steps_measured = minutes * 60, skip, None, None, 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.skip:
            self.mark = time.time()
        elapsed = time.time() - self.t0
        if elapsed > self.limit:
            print(f"\n[time budget] {elapsed/60:.1f} min reached; stopping at step {state.global_step}.")
            control.should_training_stop = True
        return control

    def seconds_per_step(self, final_step):
        if self.mark is None or final_step <= self.skip:
            return None
        return (time.time() - self.mark) / (final_step - self.skip)


@torch.inference_mode()
def predict_rows(model, tokenizer, examples, device):
    """Same windowing and span policy as qa.inference.Engine, in fp32 like CPU serving."""
    model.eval()
    rows = []
    for ex in examples:
        enc = encode(tokenizer, [ex["question"]], [ex["context"]], MAX_LEN, STRIDE)
        windows = []
        for lo in range(0, len(enc["input_ids"]), 8):
            hi = min(lo + 8, len(enc["input_ids"]))
            inputs = {k: torch.tensor(enc[k][lo:hi], device=device) for k in tokenizer.model_input_names}
            out = model(**inputs)
            starts, ends = out.start_logits.float().cpu().numpy(), out.end_logits.float().cpu().numpy()
            for j in range(hi - lo):
                i = lo + j
                offsets = [o if s == 1 else None for o, s in zip(enc["offset_mapping"][i], enc.sequence_ids(i))]
                windows.append((starts[j], ends[j], offsets, enc["input_ids"][i].index(tokenizer.cls_token_id)))
        c = candidate(ex["context"], windows, POLICY)
        rows.append({"id": ex["id"], "references": list(ex["answers"]["text"]),
                     "text": c["text"], "start": c["start"], "margin": c["margin"]})
    return rows


def evaluate(rows, threshold, prior):
    groups = {"answerable": [], "unanswerable": []}
    for r in rows:
        prediction = "" if r["margin"] > threshold else r["text"]
        groups["answerable" if r["references"] else "unanswerable"].append(score(prediction, r["references"]))
    result = {}
    for name, values in groups.items():
        result[name] = {"count": len(values),
                        "EM": 100 * sum(v[0] for v in values) / max(len(values), 1),
                        "F1": 100 * sum(v[1] for v in values) / max(len(values), 1)}
    for key in ("EM", "F1"):
        result[f"weighted_{key}"] = prior * result["answerable"][key] + (1 - prior) * result["unanswerable"][key]
    return result


def calibrate(rows, prior):
    """Exact sweep. A row answers when margin <= threshold; objective = mean of weighted EM and F1."""
    pre = []
    for r in rows:
        span, null = score(r["text"], r["references"]), score("", r["references"])
        pre.append((r["margin"], bool(r["references"]), span, null))
    counts = {True: sum(p[1] for p in pre), False: sum(not p[1] for p in pre)}
    if not counts[True] or not counts[False]:
        raise ValueError("Calibration needs answerable and unanswerable examples")
    totals = {g: [sum(p[3][k] for p in pre if p[1] == g) for k in (0, 1)] for g in (True, False)}

    def objective():
        w = [prior * totals[True][k] / counts[True] + (1 - prior) * totals[False][k] / counts[False] for k in (0, 1)]
        return (w[0] + w[1]) / 2

    pre.sort(key=lambda p: p[0])
    best_value, best_threshold = objective(), pre[0][0] - 1.0
    for i, (margin, group, span, null) in enumerate(pre):
        for k in (0, 1):
            totals[group][k] += span[k] - null[k]
        if i + 1 < len(pre) and pre[i + 1][0] == margin:
            continue
        value = objective()
        if value > best_value:
            upper = pre[i + 1][0] if i + 1 < len(pre) else margin + 1.0
            best_value, best_threshold = value, (margin + upper) / 2
    return float(best_threshold), best_value


def split_validation(validation, n_cal, n_rep, seed):
    titles = sorted(set(validation["title"]))
    random.Random(seed).shuffle(titles)
    cal_titles = set(titles[: len(titles) // 2])
    cal = [i for i, t in enumerate(validation["title"]) if t in cal_titles]
    rep = [i for i, t in enumerate(validation["title"]) if t not in cal_titles]
    rng = random.Random(seed + 1)
    rng.shuffle(cal)
    rng.shuffle(rep)
    return validation.select(cal[:n_cal]), validation.select(rep[:n_rep])


def plot_loss(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = [(h["step"], h["loss"]) for h in history if "loss" in h]
    if not points:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*points), label="training loss")
    plt.xlabel("optimizer step")
    plt.ylabel("loss")
    plt.title("DeBERTa-v3-base full fine-tuning on SQuAD 2.0")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true", help="measure speed and recommend --train-examples")
    p.add_argument("--train-examples", type=int, default=60000)
    p.add_argument("--budget-min", type=float, default=45, help="training time budget in minutes")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--grad-ckpt", action="store_true", help="use if you get CUDA out of memory")
    p.add_argument("--calib-examples", type=int, default=1500)
    p.add_argument("--report-examples", type=int, default=1500)
    p.add_argument("--prior", type=float, default=0.75, help="answerable share of the target test set")
    p.add_argument("--output", default="models/deberta-full-fast")
    p.add_argument("--compare-dir", default=None, help="existing model folder to evaluate on the same report set")
    p.add_argument("--base", default=BASE)
    p.add_argument("--dataset", default="rajpurkar/squad_v2")
    p.add_argument("--seed", type=int, default=20261101)
    a = p.parse_args()

    set_seed(a.seed)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Install the CUDA build of torch (see instructions).")

    revision = REVISION if a.base == BASE else None
    tokenizer = AutoTokenizer.from_pretrained(a.base, revision=revision, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(a.base, revision=revision)
    model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    total = sum(x.numel() for x in model.parameters())
    print(f"Parameters: {total:,} total, {trainable:,} trainable (word embeddings frozen)")

    data = load_squad(a.dataset)
    n = min(3000 if a.probe else a.train_examples, len(data["train"]))
    train_raw = data["train"].shuffle(seed=a.seed).select(range(n))
    train = tokenize_train(train_raw, tokenizer)
    ratio = len(train) / n
    print(f"{n:,} examples -> {len(train):,} windows ({ratio:.3f} windows/example)")

    out = ROOT / a.output if not Path(a.output).is_absolute() else Path(a.output)
    args = TrainingArguments(
        output_dir=str(out / "_trainer"), per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr, weight_decay=0.01,
        warmup_ratio=0.1, lr_scheduler_type="linear", num_train_epochs=1,
        max_steps=60 if a.probe else -1, bf16=cuda, tf32=cuda or None, logging_steps=10 if a.probe else 25,
        save_strategy="no", report_to="none", dataloader_num_workers=0, group_by_length=True,
        gradient_checkpointing=a.grad_ckpt, optim="adamw_torch_fused" if cuda else "adamw_torch",
        seed=a.seed, disable_tqdm=False)
    clock = Clock(a.budget_min)
    trainer = Trainer(model=model, args=args, train_dataset=train,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
                      processing_class=tokenizer, callbacks=[clock])
    start = time.time()
    trainer.train()
    seconds = time.time() - start

    if a.probe:
        sps = clock.seconds_per_step(trainer.state.global_step)
        windows_per_second = a.batch * a.accum / sps
        fit = int(0.85 * a.budget_min * 60 * windows_per_second / ratio)
        fit = max(5000, min(SQUAD_TRAIN_SIZE, fit // 1000 * 1000))
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0
        print("\n================ PROBE RESULT ================")
        print(f"{sps:.3f} s per optimizer step | {windows_per_second:.1f} windows/s | peak GPU memory {peak:.2f} GiB")
        print(f"Recommended for a {a.budget_min:.0f}-minute training budget (15% safety margin):")
        print(f"  python -m experiments.train_fast --train-examples {fit}"
              + (f" --compare-dir models/deberta-base-student" if (ROOT / 'models/deberta-base-student').exists() else ""))
        print("==============================================")
        return

    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    history = trainer.state.log_history
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_loss(history, out / "loss.png")
    weights = out / "model.safetensors"

    cal_set, rep_set = split_validation(data["validation"], a.calib_examples, a.report_examples, a.seed)
    model = AutoModelForQuestionAnswering.from_pretrained(out).to(device).float()
    print(f"Calibrating no-answer threshold on {len(cal_set)} validation examples...")
    threshold, objective = calibrate(predict_rows(model, tokenizer, cal_set, device), a.prior)
    print(f"Evaluating on {len(rep_set)} disjoint validation examples...")
    report = evaluate(predict_rows(model, tokenizer, rep_set, device), threshold, a.prior)

    metadata = {"base_model": BASE, "source_revision": REVISION, "optimizer_steps": trainer.state.global_step,
                "max_seq_length": MAX_LEN, "doc_stride": STRIDE,
                "training_method": "full fine-tuning, word embeddings frozen, bf16 mixed precision",
                "train_examples": n, "train_windows": len(train), "epochs": round(trainer.state.epoch or 0, 3),
                "learning_rate": a.lr, "warmup_ratio": 0.1, "effective_batch": a.batch * a.accum,
                "training_seconds": round(seconds, 1), "trainable_parameters": trainable,
                "hardware": torch.cuda.get_device_name(0) if cuda else "CPU", "seed": a.seed}
    calibration = {"threshold": threshold, "objective": f"mean(weighted EM, weighted F1) at {a.prior:.0%} answerable",
                   "objective_value": objective, "policy": POLICY, "weights_sha256": sha256(weights),
                   "calibration_examples": len(cal_set), "rule": "null when null_score - best_span_score > threshold"}
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    results = {"new_model": {"dir": str(out), "threshold": threshold, **report}}
    if a.compare_dir:
        old_dir = ROOT / a.compare_dir if not Path(a.compare_dir).is_absolute() else Path(a.compare_dir)
        old_cal = old_dir / "calibration.json"
        old_threshold = (json.loads(old_cal.read_text())["threshold"] if old_cal.exists()
                         else json.loads((ROOT / "configs/active_model.json").read_text())["calibration"]["threshold"])
        old_tok = AutoTokenizer.from_pretrained(old_dir, local_files_only=True, use_fast=True)
        old_model = AutoModelForQuestionAnswering.from_pretrained(old_dir, local_files_only=True).to(device).float()
        print(f"Evaluating current model {old_dir.name} on the same examples...")
        results["current_model"] = {"dir": str(old_dir), "threshold": old_threshold,
                                    **evaluate(predict_rows(old_model, old_tok, rep_set, device), old_threshold, a.prior)}
        del old_model

    sample_path = ROOT / "originals/public_sample.json"
    if sample_path.exists():
        items = json.loads(sample_path.read_text(encoding="utf-8"))["items"]
        examples = [{"id": x["id"], "question": x["question"], "context": x["context"],
                     "answers": {"text": [r["text"] for r in x.get("answers", [])]}} for x in items]
        rows = predict_rows(model, tokenizer, examples, device)
        public = []
        for r in rows:
            answer = None if r["margin"] > threshold else r["text"]
            public.append({"id": r["id"], "answer": answer, "expected": r["references"] or None,
                           "correct": bool(score(answer or "", r["references"])[0])})
        results["public_sample"] = public
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n================ RESULTS (disjoint validation articles) ================")
    print(f"Training: {trainer.state.global_step} steps, {seconds/60:.1f} min. Threshold: {threshold:.4f}")
    for name, r in results.items():
        if name == "public_sample":
            continue
        print(f"{name:>14}: weighted EM {r['weighted_EM']:.2f} | weighted F1 {r['weighted_F1']:.2f} | "
              f"answerable EM {r['answerable']['EM']:.2f} | unanswerable EM {r['unanswerable']['EM']:.2f}")
    for item in results.get("public_sample", []):
        print(f"  {item['id']}: {item['answer']!r}  expected {item['expected']}  {'OK' if item['correct'] else 'WRONG'}")
    print(f"\nModel saved to {out}. Loss curve: {out / 'loss.png'}")
    print("Serve it with:  $env:MODEL_DIR = (Resolve-Path '" + a.output.replace('/', '\\') + "').Path")


if __name__ == "__main__":
    main()
