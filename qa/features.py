"""Shared overflow tokenization and character-to-token supervision."""
def encode(tokenizer, questions, contexts, max_length=384, stride=128):
    # Bound only the question, preserving every context token through overflow.
    questions = [tokenizer.decode(tokenizer(q.lstrip(), add_special_tokens=False)["input_ids"][:64],
                                  skip_special_tokens=True) for q in questions]
    return tokenizer(questions, contexts, truncation="only_second", max_length=max_length,
                     stride=stride, return_overflowing_tokens=True,
                     return_offsets_mapping=True, padding="max_length")


def training_features(examples, tokenizer, max_length=384, stride=128):
    batch = encode(tokenizer, examples["question"], examples["context"], max_length, stride)
    mapping = batch.pop("overflow_to_sample_mapping")
    offsets = batch.pop("offset_mapping")
    starts, ends = [], []
    for i, sample in enumerate(mapping):
        cls = batch["input_ids"][i].index(tokenizer.cls_token_id)
        context_tokens = [j for j, s in enumerate(batch.sequence_ids(i)) if s == 1]
        answer = examples["answers"][sample]
        start = end = cls
        if context_tokens and answer["answer_start"]:
            a = answer["answer_start"][0]
            b = a + len(answer["text"][0])
            first, last = context_tokens[0], context_tokens[-1]
            if offsets[i][first][0] <= a and offsets[i][last][1] >= b:
                start = next(j for j in reversed(context_tokens) if offsets[i][j][0] <= a)
                end = next(j for j in context_tokens if offsets[i][j][1] >= b)
        starts.append(start)
        ends.append(end)
    batch["start_positions"], batch["end_positions"] = starts, ends
    return dict(batch)
