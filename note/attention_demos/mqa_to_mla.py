import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# tiny dims:  MQA shared-latent dim d_c=4, MLA per-head bottleneck head_dim=2
d_model, d_c, head_dim = 6, 4, 2
rng = np.random.default_rng(0)

# MQA score = (W_Q h_t) . (W_K h_i) = (W_Q h_t) . c_i     [c_i = shared latent, dim d_c]
# MLA needs:  W_Q (d_c x d_model)  =  W_UK^T (d_c x head_dim) · W_Q_mla (head_dim x d_model)
#   -> a rank-<=head_dim factorization of W_Q. Exact iff rank(W_Q) <= head_dim.

def convert(WQ, label):
    U, S, Vt = np.linalg.svd(WQ, full_matrices=False)      # WQ = U diag(S) Vt
    r = head_dim
    WUK_T = U[:, :r] * S[:r]        # (d_c x head_dim)   = W_UK^T
    WQ_mla = Vt[:r]                 # (head_dim x d_model)
    WQ_approx = WUK_T @ WQ_mla      # best rank-head_dim reconstruction
    WUK = WUK_T.T                   # (head_dim x d_c)
    err = np.linalg.norm(WQ - WQ_approx)

    # sample score check on random query h_t and cached latent c_i
    ht = rng.standard_normal(d_model); ci = rng.standard_normal(d_c)
    mqa = (WQ @ ht) @ ci                    # original MQA score
    mla = (WQ_mla @ ht) @ (WUK @ ci)        # MLA (rank-head_dim) score

    print(f"\n===== {label} =====")
    print("W_Q =\n", WQ)
    print("rank(W_Q) =", np.linalg.matrix_rank(WQ), "  (head_dim bottleneck =", head_dim, ")")
    print("singular values         :", S)
    print("  kept (top head_dim)   :", S[:r])
    print("  DISCARDED by MLA      :", S[r:], " <-- this is the loss")
    print("reconstruction error ||W_Q - W_UK^T·W_Q_mla|| =", round(err, 6))
    print(f"sample score:  MQA={mqa:+.4f}   MLA={mla:+.4f}   |diff|={abs(mqa-mla):.4f}")

# ---- Case A: learned W_Q happens to be rank 2 (= head_dim)  -> LOSSLESS ----
A = rng.standard_normal((d_c, head_dim))         # 4x2
B = rng.standard_normal((head_dim, d_model))     # 2x6
WQ_lowrank = A @ B                               # rank <= 2 by construction
convert(WQ_lowrank, "Case A: rank(W_Q)=2 == head_dim  ->  LOSSLESS")

# ---- Case B: learned W_Q is full rank 4 (> head_dim)  -> LOSSY ----
WQ_full = rng.standard_normal((d_c, d_model))    # generically rank 4
convert(WQ_full, "Case B: rank(W_Q)=4 > head_dim  ->  LOSSY")
