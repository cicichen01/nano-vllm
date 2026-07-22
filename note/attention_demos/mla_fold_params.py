import numpy as np

# realistic-ish per-head dims
d_model, head_dim, d_c = 4096, 128, 512

# ---- verify the fold is mathematically exact (tiny random check) ----
rng = np.random.default_rng(0)
WQ  = rng.standard_normal((head_dim, d_model))   # d_model -> head_dim
WUK = rng.standard_normal((head_dim, d_c))       # d_c -> head_dim
h   = rng.standard_normal(d_model)

# factored path: h -> q(head_dim) -> q'(d_c)
q_factored = WUK.T @ (WQ @ h)
# folded dense path: precompute WQ' = WUK^T WQ  (d_c x d_model), then WQ' @ h
WQ_prime = WUK.T @ WQ
q_folded = WQ_prime @ h
print("fold exact?", np.allclose(q_factored, q_folded))

# ---- parameter / compute comparison (per head, query side) ----
p_factored = WQ.size + WUK.size          # keep W_Q and W_UK separate
p_folded   = WQ_prime.size               # store dense WQ'
print(f"\nquery projection, per head:")
print(f"  factored (W_Q {WQ.shape} + W_UK {WUK.shape}) = {p_factored:,} params")
print(f"  folded dense WQ' {WQ_prime.shape}            = {p_folded:,} params")
print(f"  folding INFLATES params by {p_folded/p_factored:.1f}x")

# compute (MACs) to produce q' for one token
mac_factored = head_dim*d_model + d_c*head_dim   # through the 128-dim bottleneck
mac_folded   = d_c*d_model                        # dense
print(f"\n  MACs/token factored = {mac_factored:,}")
print(f"  MACs/token folded   = {mac_folded:,}   ({mac_folded/mac_factored:.1f}x more)")
print("\nrank of folded WQ' =", np.linalg.matrix_rank(WQ_prime),
      "(<= head_dim; the dense matrix wastes its extra capacity)")
