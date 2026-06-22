"""
Step 3 demo: prefill-priority blocks decode.
We start a few 'old' requests decoding, then INJECT big new prompts mid-run and
watch the old request's inter-token latency spike (decode is paused while the new
prompt prefills) — and see the throughput effect.

  ENV=/home/cicichen/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib \
    MODEL=~/huggingface/Qwen3-0.6B $ENV/bin/python h100_setup/explore_step3_prefill_block.py
"""
import os, time


def main():
    from nanovllm import LLM, SamplingParams
    path = os.path.expanduser(os.environ.get("MODEL", "~/huggingface/Qwen3-0.6B"))
    # small token budget so a big injected prompt takes SEVERAL prefill steps (visible stall)
    llm = LLM(path, enforce_eager=True, max_num_seqs=16,
              max_num_batched_tokens=128, max_model_len=1024)
    sp = SamplingParams(temperature=0.6, max_tokens=120, ignore_eos=True)

    # 3 "old" short-prompt requests -> they prefill fast, then decode for a while
    for _ in range(3):
        llm.add_request(list(range(1, 21)), sp)        # 20-token prompt
    victim = llm.scheduler.waiting[0]                   # track the first old request

    BIG = list(range(1, 401))                           # 400-token prompt -> ~4 prefill chunks
    inject_at = {30, 60, 90}                            # inject a big new request at these steps

    timeline = []                  # 'P' (prefill, decode blocked) or 'D' (decode) per step
    victim_token_step = []         # step index at which victim emitted each token
    victim_token_time = []         # wall-clock at each victim token
    victim_prev = victim.num_completion_tokens
    decode_tokens = 0
    step_i = 0
    t0 = time.perf_counter()
    while not llm.is_finished():
        if step_i in inject_at:
            llm.add_request(list(BIG), sp)
            print(f"  >> injected a 400-token request at step {step_i}")
        _, num_tokens = llm.step()
        phase = 'P' if num_tokens > 0 else 'D'
        timeline.append(phase)
        if phase == 'D':
            decode_tokens += -num_tokens
        if victim.num_completion_tokens > victim_prev:
            victim_token_step.append(step_i)
            victim_token_time.append(time.perf_counter())
            victim_prev = victim.num_completion_tokens
        step_i += 1
    elapsed = time.perf_counter() - t0

    print("\n=== step timeline (P=prefill: decode BLOCKED, D=decode) ===")
    print("".join(timeline))

    # victim inter-token gaps in STEPS (steady-state should be 1; spikes = blocked by prefill)
    gaps = [(victim_token_step[i], victim_token_step[i] - victim_token_step[i-1])
            for i in range(1, len(victim_token_step))]
    spikes = [(at, g) for (at, g) in gaps if g > 1]
    print("\n=== victim inter-token latency (in STEPS) ===")
    print(f"  steady-state gap = 1 step.  SPIKES (gap>1, = prefill stalls): {spikes}")

    # same in wall-clock ms
    itl_ms = [(victim_token_time[i]-victim_token_time[i-1])*1000 for i in range(1, len(victim_token_time))]
    if itl_ms:
        import statistics
        med = statistics.median(itl_ms)
        worst = max(itl_ms)
        print(f"  median ITL = {med:.1f} ms   worst ITL = {worst:.1f} ms   (worst/median = {worst/med:.1f}x)")

    print(f"\n=== throughput ===")
    print(f"  steps={step_i}  decode_tokens={decode_tokens}  time={elapsed:.2f}s  "
          f"decode_throughput={decode_tokens/elapsed:.0f} tok/s")
    n_prefill = timeline.count('P'); n_decode = timeline.count('D')
    print(f"  prefill steps={n_prefill}  decode steps={n_decode}  "
          f"(during the {n_prefill} prefill steps, ALL old requests' decode was paused)")


if __name__ == "__main__":
    main()
