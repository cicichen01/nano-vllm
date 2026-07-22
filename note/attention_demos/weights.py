import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# same 3 keys as the worked example; k3 overlaps k1,k2
K = np.array([[1.,0,0],[0,1,0],[0.6,0.8,0]])
q = K[2]                                  # query = k3
scores = K @ q                            # [q·k1, q·k2, q·k3]
print("raw scores  q·kⱼ =", scores)

# --- linear attention: w = scores ---
print("\nlinear  weights wⱼ =", scores)

# --- softmax attention ---
e = np.exp(scores - scores.max()); print("softmax weights wⱼ =", e/e.sum())

# --- DeltaNet: w = scores @ (I+L)^{-1}, L = strictly-lower-tri of key Gram ---
G = K @ K.T                               # Gram matrix kⱼ·kᵢ
L = np.tril(G, -1)                        # strictly lower triangular (causal, i<j)
IL_inv = np.linalg.inv(np.eye(3) + L)
w_delta = scores @ IL_inv
print("\nL (strict-lower key-Gram) =\n", L)
print("(I+L)^{-1} =\n", IL_inv)
print("delta   weights wⱼ = scores @ (I+L)^{-1} =", w_delta)
print("\n-> delta weights subtract the overlapping keys' shares -> clean [0,0,1]")
