import numpy as np
np.set_printoptions(precision=2, suppress=True)
def sm(x): e=np.exp(x); return e/e.sum(-1,keepdims=True)

N, d_k = 6, 2                       # 6 tokens, head_dim=2
rng = np.random.default_rng(0)
Q = rng.standard_normal((N, d_k))
K = rng.standard_normal((N, d_k))
scores = Q @ K.T                    # pre-softmax attention scores, N×N

print("pre-softmax  QKᵀ  rank =", np.linalg.matrix_rank(scores), " (≤ d_k =", d_k, ")")
print("post-softmax softmax(QKᵀ) rank =", np.linalg.matrix_rank(sm(scores)), " (up to N =", N, ")")
print("\n=> softmax LIFTS a rank-≤d_k score matrix to (near-)full rank N.")
print("   linear attention keeps QKᵀ as-is: attention stays rank ≤ d_k.")
