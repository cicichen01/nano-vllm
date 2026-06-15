"""
Smoke + throughput test for nano-vllm on H100.

Usage:
    python test_h100.py                 # sanity + small bench, 1 GPU
    MODEL=~/huggingface/Qwen3-8B/ python test_h100.py
    TP=2 python test_h100.py            # tensor parallel across 2 GPUs

Env vars:
    MODEL  path to a Qwen3 checkpoint (default ~/huggingface/Qwen3-0.6B/)
    TP     tensor_parallel_size (default 1)
    EAGER  "1" to disable CUDA graphs (default 0 -> graphs on)

To run it again

  conda activate nanovllm   # or use the env python directly
  export
  LD_LIBRARY_PATH=/home/cicichen/.conda/envs/nanovllm/targets/x86_64-linux/lib:/home/cicichen/.conda/envs/nanovllm/lib
  MODEL=~/huggingface/Qwen3-0.6B /home/cicichen/.conda/envs/nanovllm/bin/python /home/cicichen/nano-vllm/test_h100.py

  /home/cicichen/fa_backup/ (the 293 MB compiled flash-attn .so) in place — it lets you rebuild the env without the
  10–25 min recompile.
"""
import os
import time
from random import randint, seed

from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser(os.environ.get("MODEL", "~/huggingface/Qwen3-0.6B/"))
    tp = int(os.environ.get("TP", "1"))
    eager = os.environ.get("EAGER", "0") == "1"
    assert os.path.isdir(path), f"model dir not found: {path}"

    print(f"== loading {path} | tp={tp} | enforce_eager={eager} ==")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=eager, tensor_parallel_size=tp, max_model_len=4096)

    # ---- 1) Correctness / sanity ----
    print("\n== sanity ==")
    prompts = ["Introduce yourself in one sentence.", "List all prime numbers within 30."]
    chat = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    outs = llm.generate(chat, SamplingParams(temperature=0.6, max_tokens=128))
    for p, o in zip(prompts, outs):
        print(f"\nPrompt: {p}\nCompletion: {o['text'].strip()!r}")

    # ---- 2) Throughput bench ----
    print("\n== throughput ==")
    seed(0)
    num_seqs = 256
    prompt_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(num_seqs)]
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024))
           for _ in range(num_seqs)]

    llm.generate(["warmup"], SamplingParams(max_tokens=1), use_tqdm=False)  # warm caches/graphs
    t = time.time()
    llm.generate(prompt_ids, sps, use_tqdm=False)
    dt = time.time() - t
    total = sum(sp.max_tokens for sp in sps)
    print(f"Total: {total} tok | Time: {dt:.2f}s | Throughput: {total / dt:.2f} tok/s")


if __name__ == "__main__":
    main()
