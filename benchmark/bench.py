import os
import json
import sys
import time
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '../inference')
encoding_dir = os.path.join(current_dir, '../encoding')
sys.path.insert(0, os.path.abspath(encoding_dir))
sys.path.insert(0, os.path.abspath(model_dir))
from model import Transformer, ModelArgs
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


def sample(logits, temperature: float = 1.0):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial."""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """Batch generation with left-padded prompts.

    The first forward pass processes [min_prompt_len:] tokens (prefill phase).
    Subsequent passes generate one token at a time (decode phase). For positions
    still within a prompt, the ground-truth token overrides the model's prediction.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    print(f"total_len: {total_len}")
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    info = {}
    is_prefill = True
    prefill_t0 = 0.0
    prefill_t1 = 0.0
    gen_t0 = 0.0
    gen_t1 = 0.0

    torch.cuda.synchronize()
    prefill_t0 = time.perf_counter()
    for cur_pos in range(min(prompt_lens), min(prompt_lens)+1):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)

        prev_pos = cur_pos

        if finished.all():
            break
    torch.cuda.synchronize()
    prefill_t1 = time.perf_counter()
    
    gen_t0 = time.perf_counter()
    for cur_pos in range(min(prompt_lens)+1, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)

        prev_pos = cur_pos

        if finished.all():
            break
    torch.cuda.synchronize()
    gen_t1 = time.perf_counter()

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens, {"prefill_t0": prefill_t0, "prefill_t1": prefill_t1, "gen_t0": gen_t0, "gen_t1": gen_t1}


def next_frontier(ctx_max: int, step_mul: float, step_incr: int, cur: int):
    INT_MAX = 2**31 - 1
    if (cur >= ctx_max):
        return ctx_max;
    next = 0
    if (step_mul == 1.0):
        if (cur > INT_MAX - step_incr):
            next = ctx_max
        else:
            next = cur + step_incr
    else:
        v = ceil(cur * step_mul)
        next = ctx_max if v > INT_MAX else int(v)
        if (next <= cur):
            next = cur + 1
    if (next > ctx_max):
        next = ctx_max
    return next


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    ctx_start: int = 2048,
    ctx_max: int = 32768,
    step_mul: int = 1,
    step_incr: int = 2048,
    gen_tokens: int = 128,
    temperature: float = 0.6,
    csv_path: str = "",
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_t0 = time.perf_counter()
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    load_t1 = time.perf_counter()
    print(f"load time: {load_t1 - load_t0:.2f} seconds")
    print("I'm DeepSeek 👋")

    with open(input_file) as f:
        prompts = f.read()
    prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompts}], thinking_mode="chat"))]

    if (csv_path):
        with open(csv_path, "w", encoding="utf-8") as out:
            out.write("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps\n")
            out.flush()
    else:
        print("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes", flush=True);

    frontier = ctx_start
    previous = 0
    while frontier < ctx_max:
        prompt_token = [prompt_tokens[0][:frontier]]
        completion_tokens, info = generate(model, prompt_token, gen_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens)
        prefill_tokens = frontier - previous
        prefill_sec = info["prefill_t1"] - info["prefill_t0"]
        gen_sec = info["gen_t1"] - info["gen_t0"]
        frontier = next_frontier(ctx_max, step_mul, step_incr, frontier)
        previous = frontier
        for prompt, completion in zip(prompt_token, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
        if csv_path:
            with open(csv_path, "a", encoding="utf-8") as out:
                out.write(f"{frontier},{prompt_token},{prefill_tokens / prefill_sec if prefill_sec > 0.0 else 0.0:.2f},{gen_tokens},{gen_tokens / gen_sec if gen_sec > 0.0 else 0.0:.2f}\n")
                out.flush()
        else:
            print(f"{frontier},{prompt_token},{prefill_tokens / prefill_sec if prefill_sec > 0.0 else 0.0:.2f},{gen_tokens},{gen_tokens / gen_sec if gen_sec > 0.0 else 0.0:.2f}", flush=True)
            print()
    dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="", required=True)
    parser.add_argument("--ctx-start", type=int, default=2048)
    parser.add_argument("--ctx-max", type=int, default=32768)
    parser.add_argument("--step-mul", type=int, default=1)
    parser.add_argument("--step-incr", type=int, default=2048)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--csv-path", type=str, default="")
    args = parser.parse_args()
    main(args.ckpt_path, args.config, args.input_file, args.ctx_start, args.ctx_max,
        args.step_mul, args.step_incr, args.gen_tokens, args.temperature, args.csv_path)
