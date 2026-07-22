import numpy as np
rng = np.random.default_rng(0)

# ---- Q1: does the causal mask raise the attention-matrix rank? ----
N, d_k = 8, 3
Q = rng.standard_normal((N,d_k)); K = rng.standard_normal((N,d_k))
scores = Q @ K.T
L = np.tril(np.ones((N,N)))                       # causal (lower-tri) mask
masked_linear = scores * L                        # linear attn: masked scores (Hadamard)
sc = np.where(L>0, scores, -1e9)
e = np.exp(sc - sc.max(1,keepdims=True)); soft = e/e.sum(1,keepdims=True)
print("Q1  attention-matrix rank (N=8, d_k=3):")
print("  QKᵀ  unmasked           =", np.linalg.matrix_rank(scores), " (≤ d_k=3)")
print("  QKᵀ ⊙ causal mask       =", np.linalg.matrix_rank(masked_linear), " <- mask RAISES it")
print("  softmax(masked QKᵀ)     =", np.linalg.matrix_rank(soft))

# ---- Q4: is rank(S) higher for delta than plain? ----
dk=dv=16; M=40
K2 = rng.standard_normal((M,dk)); K2 /= np.linalg.norm(K2,axis=1,keepdims=True)
V2 = rng.standard_normal((M,dv))
Sp = np.zeros((dk,dv)); Sd = np.zeros((dk,dv))
for k,v in zip(K2,V2):
    Sp = Sp + np.outer(k,v)
    Sd = Sd + np.outer(k, v - k@Sd)
print("\nQ4  rank of the state S (dk=dv=16, M=40 items):")
print("  plain  S = Σkvᵀ  rank =", np.linalg.matrix_rank(Sp))
print("  delta  S         rank =", np.linalg.matrix_rank(Sd))
print("  -> both already FULL rank d_k; delta does NOT raise rank")
