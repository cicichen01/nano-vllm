"""
Step 4 live demo: watch the BlockManager build a block_table, register prefix-cache
hashes, then REUSE them for a second request that shares the prefix.

Two requests share their first 256 tokens (one full block_size=256 block). Run sequentially:
  req1 -> no cache yet -> allocates fresh blocks -> hash_blocks registers block 0
  req1 finishes -> deallocate (block 0 becomes free-but-still-cached)
  req2 -> can_allocate finds block 0 cached -> allocate REUSES it (prefix-cache hit)

Run:
  ENV=/home/cicichen/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib \
    MODEL=~/huggingface/Qwen3-0.6B $ENV/bin/python h100_setup/explore_step4_blocks.py
"""
import os


def main():
    from nanovllm import LLM, SamplingParams
    path = os.path.expanduser(os.environ.get("MODEL", "~/huggingface/Qwen3-0.6B"))
    llm = LLM(path, enforce_eager=True, max_model_len=1024)
    bm = llm.scheduler.block_manager
    bs = bm.block_size
    print(f"block_size = {bs}\n")

    # instrument the BlockManager (rank-0, in-process)
    o_can, o_alloc, o_hash, o_dealloc = bm.can_allocate, bm.allocate, bm.hash_blocks, bm.deallocate
    def pool(): return f"free={len(bm.free_block_ids)} used={len(bm.used_block_ids)} cache_entries={len(bm.hash_to_block_id)}"
    def can_allocate(seq):
        r = o_can(seq)
        print(f"  can_allocate(seq{seq.seq_id}): num_blocks={seq.num_blocks} -> cached_blocks={r}   [{pool()}]")
        return r
    def allocate(seq, n):
        o_alloc(seq, n)
        refs = [bm.blocks[b].ref_count for b in seq.block_table]
        print(f"  allocate(seq{seq.seq_id}, cached={n}): block_table={seq.block_table} ref_counts={refs} num_cached_tokens={seq.num_cached_tokens}")
    def hash_blocks(seq):
        before = len(bm.hash_to_block_id)
        o_hash(seq)
        after = len(bm.hash_to_block_id)
        if after > before:
            print(f"  hash_blocks(seq{seq.seq_id}): registered {after-before} completed block(s) -> {pool()}")
    def deallocate(seq):
        bt = list(seq.block_table)
        o_dealloc(seq)
        print(f"  deallocate(seq{seq.seq_id}): freed table {bt} -> {pool()}")
    bm.can_allocate, bm.allocate, bm.hash_blocks, bm.deallocate = can_allocate, allocate, hash_blocks, deallocate

    sp = SamplingParams(temperature=0.6, max_tokens=3, ignore_eos=True)
    shared = list(range(256))                      # exactly one full block (block 0)
    p1 = shared + list(range(256, 300))            # block1 = tokens 256..299
    p2 = shared + list(range(900, 944))            # SAME block 0, DIFFERENT block 1

    print(f"=== REQUEST 1 (prompt {len(p1)} tokens; block0=shared, block1=unique) ===")
    llm.generate([p1], sp, use_tqdm=False)
    print(f"  -> after req1 finishes: {pool()}  (block0 freed but STILL cached)\n")

    print(f"=== REQUEST 2 (prompt {len(p2)} tokens; block0 SAME as req1, block1 different) ===")
    llm.generate([p2], sp, use_tqdm=False)
    print(f"\n  KEY: req2's can_allocate should report cached_blocks=1 (prefix HIT on block 0),")
    print(f"       and allocate reuses that block_id instead of computing its KV again.")


if __name__ == "__main__":
    main()
