"""Single-turn rollouts with verifiers environments: prompts in, rewards out."""
import asyncio
import importlib

import numpy as np
import verifiers as vf

LOCAL_ENVS = {"countdown": "envs.countdown", "intellect-math": "envs.intellect_math"}


def load_env(env_id, **args):
    """Local environments by name; anything else from the verifiers hub."""
    if env_id in LOCAL_ENVS:
        return importlib.import_module(LOCAL_ENVS[env_id]).load_environment(**args)
    return vf.load_environment(env_id, **args)


def tokenize_prompt(messages, tokenizer, prompt_format):
    """chat: the chat template plus generation prompt (post-trained models).
    text: the message contents as a plain continuation (Base models)."""
    if prompt_format == "chat":
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=False)
    assert prompt_format == "text", prompt_format
    text = "\n\n".join(message["content"] for message in messages)
    return tokenizer.encode(text if text.endswith("\n") else text + "\n", add_special_tokens=False)


def prompt_buffer(rows, tokenizer, seq_len, prompt_format):
    """[rows, seq_len] token buffer holding each prompt, padded on the right."""
    tokens = np.full((len(rows), seq_len), tokenizer.pad_token_id, np.int32)
    prompt_lengths = np.zeros(len(rows), np.int32)
    for i, row in enumerate(rows):
        prompt = tokenize_prompt(row["prompt"], tokenizer, prompt_format)[:seq_len]
        tokens[i, :len(prompt)] = prompt
        prompt_lengths[i] = len(prompt)
    return tokens, prompt_lengths


def completions(tokens, prompt_lengths, tokenizer):
    """Loss mask over each completion (including its EOS), lengths, and texts."""
    mask = np.zeros(tokens.shape, bool)
    lengths = np.zeros(len(tokens), np.int32)
    texts = []
    for i, prompt_length in enumerate(prompt_lengths):
        generated = tokens[i, prompt_length:]
        stop = np.isin(generated, [tokenizer.eos_token_id, tokenizer.pad_token_id])
        end = int(stop.argmax()) if stop.any() else len(generated)
        lengths[i] = end + (end < len(generated) and generated[end] == tokenizer.eos_token_id)
        mask[i, prompt_length:prompt_length + lengths[i]] = True
        texts.append(tokenizer.decode(generated[:end], skip_special_tokens=True))
    return mask, lengths, texts


def score(rubric, rows, texts, group_size):
    states = [vf.State(input=row, prompt=row["prompt"], trajectory=[], completion=[vf.AssistantMessage(content=text)],
                       answer=row.get("answer", ""), info=row.get("info", {}))
              for row, text in zip(rows, texts)]

    async def score_all():
        if rubric.has_group_rewards:
            for start in range(0, len(states), group_size):
                await rubric.score_group(states[start:start + group_size])
        else:
            await asyncio.gather(*(rubric.score_rollout(state) for state in states))
        await asyncio.gather(*(rubric.cleanup(state) for state in states))

    asyncio.run(score_all())
    return np.asarray([state.get("reward") or 0.0 for state in states], np.float32)
