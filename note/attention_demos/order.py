import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
def sm(x): e=np.exp(x); return e/e.sum()

# demo collision: token1 & token3 share key A=[1,0]; token2 is B=[0,1]
K = np.array([[1.,0],[0,1],[1,0]])          # k_A, k_B, k_A
V = np.array([[1.,0],[0,1],[0,1]])          # v_A1, v_B, v_A2
q = np.array([1.,0])                         # query for key A

print("LINEAR attention (no softmax) — try both association orders:")
y1 = (q @ K.T) @ V                           # (qKᵀ) first: keep per-token scores
y2 = q @ (K.T @ V)                           # (KᵀV)=S first: compress to state
print("  (qKᵀ)V  [scores first] =", y1)
print("  q(KᵀV)  [state S first] =", y2)
print("  identical? ", np.allclose(y1, y2), "  <-- ORDER DOESN'T MATTER, both interfere\n")

print("SOFTMAX attention — softmax sits BETWEEN qKᵀ and V (can't reassociate):")
scores = q @ K.T
y_soft = sm(scores) @ V
print("  scores =", scores, " softmax =", sm(scores))
print("  softmax(qKᵀ)V =", y_soft, "  <-- different from linear (nonlinearity),")
print("                              but STILL blends the two identical-key tokens")
