import numpy as np
rng = np.random.default_rng(0)
d_k = d_v = 4                      # small state
S = np.zeros((d_k, d_v))
print(f"d_k = d_v = {d_k};  add one random token at a time, S += outer(k,v)\n")
print(f"{'#tokens':>8} {'rank(S)':>8}")
for n in range(1, 9):
    k = rng.standard_normal(d_k); v = rng.standard_normal(d_v)
    S = S + np.outer(k, v)
    print(f"{n:>8} {np.linalg.matrix_rank(S):>8}")

# collision case: repeat the SAME key twice
print("\ncollision: two tokens with the SAME key k (different v):")
k = rng.standard_normal(d_k)
S2 = np.outer(k, rng.standard_normal(d_v)) + np.outer(k, rng.standard_normal(d_v))
print("  rank =", np.linalg.matrix_rank(S2), " (still 1: S = k·(v1+v2)ᵀ)")
