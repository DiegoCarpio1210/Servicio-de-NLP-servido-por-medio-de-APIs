# Primary sources

Original assignment: `originals/` (unaltered contents); YAML overrides the PDF summary.
Model and dataset cards inspected on 2026-10-25:
- https://huggingface.co/distilbert/distilbert-base-uncased (BASE masked-language encoder, not QA fine-tuned).
- https://huggingface.co/google-bert/bert-base-uncased (architectural alternative, no empirical comparison claimed).
- https://huggingface.co/datasets/rajpurkar/squad_v2 (official train and validation).
- https://arxiv.org/abs/1810.04805 (BERT; study sections 4.2-4.3 before defense).
- https://arxiv.org/abs/1806.03822 (adversarial unanswerable questions).
- https://arxiv.org/abs/1706.03762 (Transformer attention).
- https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf (GPT-1 investigation).
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/ (temporary HTTPS tunnel; no account).
- https://ngrok.com/docs/start (ngrok setup documentation).

No results from these papers or model cards are reported as our measurements.

Continuation and calibration methods:
- https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py (official normalization and reference filtering).
- https://github.com/huggingface/transformers/blob/v4.57.1/examples/pytorch/question-answering/utils_qa.py (window/span postprocessing).
- https://huggingface.co/docs/peft/main/en/conceptual_guides/lora (adapter alternative considered, not implemented or benchmarked).
