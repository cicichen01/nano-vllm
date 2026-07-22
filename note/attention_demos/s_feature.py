import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# two NON-orthogonal unit keys (realistic: related tokens -> similar keys)
k1 = np.array([1.0, 0.0]); v1 = np.array([1.0, 0.0])
k2 = np.array([0.8, 0.6]); v2 = np.array([0.0, 1.0])
K = np.stack([k1, k2]); V = np.stack([v1, v2])

print("keys not orthogonal:  k1·k2 =", k1 @ k2, "\n")

# (a) plain linear attention: S = Σ k vᵀ  (correlation / Hebbian memory)
S_plain = np.outer(k1, v1) + np.outer(k2, v2)
# (b) ideal associative map: regression / pseudoinverse solution
S_star = np.linalg.pinv(K) @ V
# (c) delta rule, one causal pass (β=1): error-correcting online regression
S_delta = np.zeros((2, 2))
for k, v in [(k1, v1), (k2, v2)]:
    S_delta = S_delta + np.outer(k, v - k @ S_delta)

for name, S in [("plain  Σkvᵀ", S_plain), ("ideal  K⁺V", S_star), ("delta rule", S_delta)]:
    print(f"{name}:  read k1 -> {k1@S}   read k2 -> {k2@S}   (want [1,0] and [0,1])")
