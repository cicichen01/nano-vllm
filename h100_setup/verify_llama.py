"""Correctness oracle for the Llama port: nano-vllm (greedy) vs HF transformers (greedy).
nano-vllm forbids greedy + uses stochastic Gumbel sampling, so we monkeypatch the Sampler to argmax,
run both on the SAME input_ids, and compare token-for-token.

Usage:
  ENV=~/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib \
    $ENV/bin/python h100_setup/verify_llama.py ~/huggingface/TinyLlama-1.1B-Chat-v1.0
"""
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/huggingface/TinyLlama-1.1B-Chat-v1.0")
prompt = "The capital of France is"
N = 24

tok = AutoTokenizer.from_pretrained(path)
ids = tok(prompt, return_tensors="pt").input_ids                       # [1, L]

# ---- HF reference (greedy) ----
hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    ref = hf.generate(ids.cuda(), do_sample=False, max_new_tokens=N)
ref_new = ref[0, ids.shape[1]:].tolist()
del hf; torch.cuda.empty_cache()

# ---- nano-vllm (patch sampler → greedy, so it's deterministic) ----
from nanovllm.layers import sampler
def greedy(self, logits, temperatures):                                # argmax = greedy (ignores temp/Gumbel)
    return logits.float().argmax(dim=-1)
sampler.Sampler.forward = greedy

from nanovllm import LLM, SamplingParams
llm = LLM(path, enforce_eager=True, max_model_len=2048)
out = llm.generate([ids[0].tolist()], SamplingParams(temperature=1.0, max_tokens=N, ignore_eos=True), use_tqdm=False)
nv_new = out[0]["token_ids"]

# ---- compare ----
k = min(len(ref_new), len(nv_new))
match = 0
for a, b in zip(ref_new, nv_new):
    if a != b: break
    match += 1
print("\n=== correctness: nano-vllm(greedy) vs HF(greedy) ===")
print("HF   token ids:", ref_new)
print("nano token ids:", nv_new)
print(f"leading exact-match: {match}/{k}   ({'PASS ✓' if match >= k else 'diverges after '+str(match)+' (check FP vs bug)'})")
print("HF   text:", repr(tok.decode(ref_new)))
print("nano text:", repr(tok.decode(nv_new)))
