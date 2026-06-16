"""
Step 1/2 extension: watch the TENSOR-PARALLEL (tp=2) machinery.
MUST be guarded by __main__ (mp 'spawn' re-imports this file in the worker).

Run:
  ENV=/home/cicichen/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib \
    MODEL=~/huggingface/Qwen3-0.6B $ENV/bin/python h100_setup/explore_step1_tp2.py
"""
import os, pickle


def main():
    from nanovllm import LLM, SamplingParams

    path = os.path.expanduser(os.environ.get("MODEL", "~/huggingface/Qwen3-0.6B"))
    print(f"main process PID = {os.getpid()}")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2,
              max_num_batched_tokens=4, max_model_len=512)

    # --- evidence of the 2-process, sharded-weights architecture ---
    print("\n### TP=2 ARCHITECTURE ###")
    print(f"  worker processes spawned : {len(llm.ps)}  (ranks 1..N-1)")
    for p in llm.ps:
        print(f"    worker PID={p.pid} alive={p.is_alive()}")
    mr = llm.model_runner
    print(f"  rank-0 shared-memory buf : name={mr.shm.name!r} size={mr.shm.size} bytes")
    # weight sharding: qkv_proj output dim is split across the 2 GPUs
    qkv_w = llm.model_runner.model.model.layers[0].self_attn.qkv_proj.weight
    print(f"  layer0 qkv_proj weight   : {tuple(qkv_w.shape)}  (output dim halved vs tp=1)")

    # --- instrument rank-0 -> worker IPC (write_shm) ---
    orig_write = mr.write_shm
    def write_shm(method_name, *args):
        nbytes = len(pickle.dumps([method_name, *args]))
        print(f"    [IPC] rank0 -> worker: method={method_name!r}  pickled_payload={nbytes} bytes")
        return orig_write(method_name, *args)
    mr.write_shm = write_shm

    # --- instrument rank-0 scheduler field transitions ---
    sched = llm.scheduler
    orig_schedule, orig_post = sched.schedule, sched.postprocess
    step = {"n": 0}
    def fmt(s):
        return (f"num_tokens={s.num_tokens:<3} cached={s.num_cached_tokens:<3} "
                f"scheduled={s.num_scheduled_tokens:<2} is_prefill={str(s.is_prefill):<5} completion={s.num_completion_tokens}")
    def schedule():
        seqs, is_prefill = orig_schedule()
        step["n"] += 1
        print(f"\n===== STEP {step['n']} [{'PREFILL' if is_prefill else 'DECODE'}] =====")
        for s in seqs:
            print(f"  after schedule(): {fmt(s)}")
        return seqs, is_prefill
    def postprocess(seqs, token_ids, is_prefill):
        orig_post(seqs, token_ids, is_prefill)
        for s in seqs:
            print(f"  after postprocess(): {fmt(s)}")
    sched.schedule, sched.postprocess = schedule, postprocess

    print("\n### RUNNING (10-token prompt, chunked prefill 4+4+2, then decode) ###")
    llm.add_request(list(range(1, 11)), SamplingParams(temperature=0.6, max_tokens=3, ignore_eos=True))
    while not llm.is_finished():
        llm.step()
    print("\nDONE")


if __name__ == "__main__":     # CRITICAL: mp 'spawn' re-imports this file in the worker
    main()
