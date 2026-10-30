import hashlib
import json
import math
import os
from pathlib import Path
from threading import Lock
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from qa.features import encode
from qa.spans import aggregate, decide
from qa.postprocessing import candidate as policy_candidate


class Engine:
    def __init__(self, model_dir, require_calibration=True):
        root = Path(model_dir)
        with (root / "model.safetensors").open("rb") as stream:
            self.weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = json.loads((root / "training_metadata.json").read_text())
        if self.metadata.get("optimizer_steps", 0) <= 0:
            raise ValueError("A genuinely fine-tuned checkpoint is required")
        self.max_length = self.metadata["max_seq_length"]
        self.stride = self.metadata["doc_stride"]
        calibration = root / "calibration.json"
        if require_calibration and not calibration.exists():
            raise ValueError("Validation calibration is required before serving")
        self.calibration = json.loads(calibration.read_text()) if calibration.exists() else {}
        self.policy = self.calibration.get('policy')
        if self.calibration.get('weights_sha256', self.weights_sha256) != self.weights_sha256:
            raise ValueError('Calibration belongs to different weights')
        self.threshold = float(self.calibration.get('threshold',0.))
        if not math.isfinite(self.threshold): raise ValueError('Calibration threshold must be finite')
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "4")))
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(root, local_files_only=True).to("cpu").eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.lock = Lock()

    def candidate(self, context, question):
        batch = encode(self.tokenizer, [question], [context], self.max_length, self.stride)
        windows = []
        with self.lock, torch.inference_mode():
            for lo in range(0, len(batch["input_ids"]), 8):
                hi = min(lo + 8, len(batch["input_ids"]))
                inputs = {k: torch.tensor(batch[k][lo:hi]) for k in self.tokenizer.model_input_names}
                output = self.model(**inputs)
                for j in range(hi-lo):
                    i = lo+j
                    offsets = [o if s == 1 else None for o, s in
                               zip(batch["offset_mapping"][i], batch.sequence_ids(i))]
                    cls = batch["input_ids"][i].index(self.tokenizer.cls_token_id)
                    windows.append((output.start_logits[j].numpy(), output.end_logits[j].numpy(), offsets, cls))
        return policy_candidate(context, windows, self.policy) if self.policy else aggregate(context, windows)

    def predict(self, context, question):
        result = decide(self.candidate(context, question), self.threshold)
        if result["answer"] is not None:
            assert context[result["start"]:result["start"]+len(result["answer"])] == result["answer"]
        return result
