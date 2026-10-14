"""Data and logprob helpers for train_lm.py and train_dpo.py."""
import functools

import datasets
import jax
import jax.numpy as jnp
import numpy as np


def logprobs(forward, weights, tokens, mask):
    """log p of every next token, and the loss mask shifted to match."""
    targets = jnp.roll(tokens, -1, axis=1)
    mask = jnp.roll(mask, -1, axis=1).at[:, -1].set(False)
    logp = jax.nn.log_softmax(forward(tokens, weights).astype(jnp.float32))
    # one-hot rather than take_along_axis: works with an explicitly sharded vocab
    return (logp * jax.nn.one_hot(targets, logp.shape[-1])).sum(-1), mask


def _chat(tokenizer, messages, **kwargs):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, enable_thinking=False, **kwargs)


@functools.cache
def _assistant_markers(tokenizer):
    """Token prefixes that open an assistant turn, and the turn-end token, found
    by rendering an assistant turn with content '' vs 'X' and seeing where they
    diverge. The boundary token before the content can merge with it under BPE,
    so it is excluded from the prefix and skipped when matching."""
    empty = _chat(tokenizer, [{"role": "assistant", "content": ""}])
    probe = _chat(tokenizer, [{"role": "assistant", "content": "X"}])
    split = next(i for i, (a, b) in enumerate(zip(empty, probe)) if a != b)
    # Some templates (Llama 3) inject a system turn: start after its last EOS.
    start = max((i + 1 for i in range(split - 1) if empty[i] == tokenizer.eos_token_id),
                default=int(empty[0] == tokenizer.bos_token_id))
    user = [{"role": "user", "content": ""}]
    base, gen = _chat(tokenizer, user), _chat(tokenizer, user, add_generation_prompt=True)
    gen_split = next((i for i, (a, b) in enumerate(zip(base, gen)) if a != b), len(base))
    prefixes = {tuple(empty[start:split - 1]), tuple(gen[gen_split:-1])} - {()}
    return sorted(prefixes, key=len, reverse=True), empty[split]


def tokenize_chat(tokenizer, messages):
    """Chat-template tokens and a mask that is True on assistant turns."""
    tokens = _chat(tokenizer, messages)
    prefixes, end = _assistant_markers(tokenizer)
    mask = np.zeros(len(tokens), bool)
    i = 0
    while i < len(tokens):
        prefix = next((p for p in prefixes if tuple(tokens[i:i + len(p)]) == p), None)
        if prefix is not None:
            i += len(prefix) + 1
            while i < len(tokens):
                mask[i] = True
                if tokens[i] == end:
                    break
                i += 1
        i += 1
    return tokens, mask


def _pad(tokens, mask, seq_len):
    if len(tokens) > seq_len or not np.any(mask):
        return None
    padded = np.zeros(seq_len, np.int32), np.zeros(seq_len, bool)
    padded[0][:len(tokens)], padded[1][:len(tokens)] = tokens, mask
    return {"tokens": padded[0], "mask": padded[1]}


def encode(row, tokenizer, seq_len, format):
    """One dataset row -> padded arrays, or None to skip (too long / empty)."""
    if format == "text":
        tokens = tokenizer.encode(row["text"])
        return _pad(tokens, np.ones(len(tokens), bool), seq_len)
    if format == "chat":
        if any(m["content"] is None for m in row["messages"]):
            return None
        messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
        return _pad(*tokenize_chat(tokenizer, messages), seq_len)
    if format == "preference":  # columns 'chosen' / 'rejected', both message lists
        pair = [encode({"messages": row[k]}, tokenizer, seq_len, "chat")
                for k in ("chosen", "rejected")]
        if None in pair:
            return None
        return {f"{k}_{name}": v for k, ex in zip(("chosen", "rejected"), pair)
                for name, v in ex.items()}
    raise ValueError(f"unknown data.format {format!r}")


def load_batches(data, tokenizer, seed):
    """HF dataset -> (make_train(epoch) batch iterator, list of valid batches).
    The first data.valid_rows rows of a seeded shuffle are held out."""
    ds = datasets.load_dataset(data.path, data.name, split=data.split).shuffle(seed=seed)
    valid, train = ds.select(range(data.valid_rows)), ds.select(range(data.valid_rows, len(ds)))

    def batches(rows):
        batch = []
        for row in rows:
            example = encode(row, tokenizer, data.seq_len, data.format)
            if example is not None:
                batch.append(example)
            if len(batch) == data.batch_size:
                yield {k: np.stack([ex[k] for ex in batch]) for k in batch[0]}
                batch = []

    return (lambda epoch: batches(train.shuffle(seed=seed + epoch))), list(batches(valid))
