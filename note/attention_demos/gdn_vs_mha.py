import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
def sm(x): e=np.exp(x); return e/e.sum()

# 2 past tokens + a query, per head
q  = np.array([1.,0.])
k1,v1 = np.array([1.,0.]), np.array([2.,0.])
k2,v2 = np.array([0.,1.]), np.array([0.,3.])

# --- MHA: softmax over per-token scores, then weighted sum ---
scores = np.array([q@k1, q@k2])          # q·k_i  for each stored token
a = sm(scores)                            # softmax normalizes across tokens
y_mha = a[0]*v1 + a[1]*v2
print("MHA :  scores", scores, " softmax", a, " -> y =", y_mha)
print("       (needs BOTH k1,v1 and k2,v2 kept around; recomputed every step)")

# --- linear attention / GDN readout: accumulate state, read with q ---
S = np.outer(k1,v1) + np.outer(k2,v2)    # state = Σ k_i v_iᵀ   (folds tokens in)
y_lin = q @ S                             # y = qᵀS = Σ (q·k_i) v_i   (NO softmax)
print("\nlin :  S =\n", S)
print("       y = qᵀS =", y_lin, "  (individual k1,v1,k2,v2 no longer needed)")
