"""
Step 1 explainer: watch a Sequence's state evolve through its lifecycle.
CPU-only, no model load. Run:  python h100_setup/explore_step1_sequence.py
"""
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

# Use a tiny block_size so we can SEE blocks fill up (real default is 256).
Sequence.block_size = 4

def show(seq, label):
    print(f"\n--- {label} ---")
    print(f"  status               = {seq.status.name}")
    print(f"  token_ids            = {seq.token_ids}")
    print(f"  num_tokens           = {seq.num_tokens}   (len of token_ids)")
    print(f"  num_prompt_tokens    = {seq.num_prompt_tokens}")
    print(f"  num_cached_tokens    = {seq.num_cached_tokens}   (KV already computed)")
    print(f"  num_scheduled_tokens = {seq.num_scheduled_tokens}  (planned for this step)")
    print(f"  is_prefill           = {seq.is_prefill}")
    print(f"  num_blocks           = {seq.num_blocks}   (ceil({seq.num_tokens}/{seq.block_size}))")
    print(f"  last_block_num_tokens= {seq.last_block_num_tokens}  (fill of last block)")
    print(f"  num_completion_tokens= {seq.num_completion_tokens}")

# ---- create a request: a 6-token prompt ----
prompt = [101, 102, 103, 104, 105, 106, 107]
seq = Sequence(prompt, SamplingParams(temperature=0.6, max_tokens=3))
print(f"seq_id = {seq.seq_id}  | block_size = {Sequence.block_size}")
show(seq, "born: WAITING, nothing cached yet")

# ---- simulate CHUNKED PREFILL: process 4 of 6 prompt tokens this step ----
seq.num_scheduled_tokens = max(4, seq.num_tokens - seq.num_cached_tokens)
show(seq, "scheduled a 4-token prefill chunk")
seq.num_cached_tokens += seq.num_scheduled_tokens   # what postprocess() does
seq.num_scheduled_tokens = 0
print("\n>>> after step: num_cached_tokens(4) < num_tokens(6) -> STILL prefilling")

# ---- finish prefill: process remaining 2 tokens ----
seq.num_scheduled_tokens = max(4, seq.num_tokens - seq.num_cached_tokens)
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
seq.is_prefill = False
seq.status = SequenceStatus.RUNNING
print(">>> now num_cached_tokens(6) == num_tokens(6) -> prefill DONE, enter decode")
show(seq, "prefill complete, RUNNING")

# ---- DECODE: each step appends exactly 1 sampled token ----
for tok in (501, 502):
    seq.append_token(tok)          # the ONLY way a sequence grows
    show(seq, f"decoded token {tok}  (watch num_blocks/last_block grow)")

print("\nKEY TAKEAWAYS:")
print("  * num_cached_tokens < num_tokens  => still prefilling")
print("  * decode appends 1 token/step; num_blocks grows when last block overflows")
print("  * everything (scheduler, block mgr, runner) just reads/writes these fields")


# ============================================================================
# REAL ENGINE TRACE  (needs GPU + model)
#   REAL=1 MODEL=~/huggingface/Qwen3-0.6B \
#     LD_LIBRARY_PATH=<env>/targets/x86_64-linux/lib:<env>/lib \
#     python h100_setup/explore_step1_sequence.py
# A tiny max_num_batched_tokens forces the prompt to be prefilled in CHUNKS so
# you can watch num_cached / num_scheduled / num_tokens move across real steps.
# ============================================================================
import os

def real_engine_trace():
    from nanovllm import LLM, SamplingParams as SP        # heavy import (flash_attn) -> lazy
    path = os.path.expanduser(os.environ.get("MODEL", "~/huggingface/Qwen3-0.6B"))
    llm = LLM(path, enforce_eager=True, max_num_batched_tokens=4, max_model_len=512)
    sched = llm.scheduler
    orig_schedule, orig_post = sched.schedule, sched.postprocess
    step = {"n": 0}

    def fmt(s):
        return (f"status={s.status.name:<8} num_tokens={s.num_tokens:<3} cached={s.num_cached_tokens:<3} "
                f"scheduled={s.num_scheduled_tokens:<2} is_prefill={str(s.is_prefill):<5} "
                f"blocks={s.num_blocks} completion={s.num_completion_tokens}")

    def schedule():
        seqs, is_prefill = orig_schedule()
        step["n"] += 1
        print(f"\n===== STEP {step['n']}  [{'PREFILL' if is_prefill else 'DECODE'}] =====")
        for s in seqs:
            print(f"  after schedule():          {fmt(s)}")
        return seqs, is_prefill

    def postprocess(seqs, token_ids, is_prefill):
        # decide BEFORE postprocess whether this prefill chunk is partial
        partial = [is_prefill and (s.num_cached_tokens + s.num_scheduled_tokens < s.num_tokens) for s in seqs]
        orig_post(seqs, token_ids, is_prefill)
        for s, t, p in zip(seqs, token_ids, partial):
            tag = "(partial chunk -> token DISCARDED)" if p else "(token APPENDED)"
            print(f"  after run()+postprocess(): {fmt(s)}  sampled={t} {tag}")

    sched.schedule, sched.postprocess = schedule, postprocess

    prompt = list(range(1, 11))          # 10-token prompt
    print(f"PROMPT = {len(prompt)} tokens | max_num_batched_tokens=4 -> forces chunked prefill (4+4+2)")
    llm.add_request(prompt, SP(temperature=0.6, max_tokens=3, ignore_eos=True))
    while not llm.is_finished():
        llm.step()
    print("\nDONE")

if os.environ.get("REAL") == "1":
    real_engine_trace()
