"""
Tensor-parallel building blocks on REAL 2-GPU NCCL: Column-parallel vs Row-parallel,
and what goes wrong if you forget the collective.

A linear is y = X @ W.T  (X[M,K], W[N,K], y[M,N]).
  COLUMN-parallel: split the OUTPUT dim N -> each rank computes an output SLICE,
                   combine = all_gather (concat). NO sum needed.
  ROW-parallel:    split the contraction dim K -> each rank computes a PARTIAL sum,
                   combine = all_reduce (sum). Forget it -> wrong (only a partial).

Run:
  ENV=/home/cicichen/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib $ENV/bin/python h100_setup/tp_parallel_demo.py
"""
import torch, torch.distributed as dist, torch.multiprocessing as mp, torch.nn.functional as F
M, K, N = 4, 8, 6

def make():
    torch.manual_seed(0)
    return torch.randn(M, K), torch.randn(N, K)   # X[M,K], W[N,K]

def worker(rank, ws):
    dist.init_process_group("nccl", init_method="tcp://localhost:29556", world_size=ws, rank=rank)
    torch.cuda.set_device(rank)
    X, W = make()
    ref = F.linear(X, W).cuda()                          # single-GPU reference [M,N]

    # ---- COLUMN parallel: split OUTPUT dim N (rows of W since W is [N,K]) ----
    osh = N // ws
    Wc = W[rank*osh:(rank+1)*osh, :].cuda()              # [N/ws, K]  this rank's output rows
    Yc = F.linear(X.cuda(), Wc)                          # [M, N/ws]  output SHARD, NO comm
    gathered = [torch.empty_like(Yc) for _ in range(ws)]
    dist.all_gather(gathered, Yc)                        # combine = concat along N
    col_full = torch.cat(gathered, dim=1)                # [M, N]

    # ---- ROW parallel: split contraction dim K ----
    ish = K // ws
    Wr = W[:, rank*ish:(rank+1)*ish].cuda()              # [N, K/ws]
    Xr = X[:, rank*ish:(rank+1)*ish].cuda()              # [M, K/ws]
    Pr = F.linear(Xr, Wr)                                # [M,N] PARTIAL sum
    row_correct = Pr.clone(); dist.all_reduce(row_correct)   # combine = SUM
    row_wrong = Pr                                       # BUG: forgot all_reduce

    if rank == 0:
        print("COLUMN-parallel (split OUTPUT dim; combine = all_gather/concat):")
        print(f"   each rank holds only a {osh}-col slice; gathered == ref : {torch.allclose(col_full, ref, atol=1e-3)}")
        print("ROW-parallel (split CONTRACTION dim; combine = all_reduce/sum):")
        print(f"   WITH all_reduce   == ref : {torch.allclose(row_correct, ref, atol=1e-3)}")
        print(f"   WITHOUT all_reduce == ref : {torch.allclose(row_wrong, ref, atol=1e-3)}   max_err={(row_wrong-ref).abs().max():.2f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    print("Tensor parallelism on 2 GPUs (real NCCL):\n")
    mp.spawn(worker, args=(2,), nprocs=2)
