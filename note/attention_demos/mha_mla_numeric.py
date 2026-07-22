import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

def sm(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

# ---- tiny dims (readable) ----
d_model, n_heads, head_dim, d_c = 4, 2, 2, 2
scale = 1/np.sqrt(head_dim)

# 3 context tokens (rows) in the residual stream (d_model=4)
H = np.array([[1., 0., 1., 0.],
              [0., 1., 1., 0.],
              [1., 1., 0., 1.]])
q_idx = 2                      # we decode/attend FROM token 2 (the last one)

print("="*70); print("SHARED INPUT  H  (3 tokens x d_model=4)"); print("="*70)
print(H)

# ======================================================================
# (1) FULL MHA
# ======================================================================
print("\n"+"="*70); print("(1) FULL MHA"); print("="*70)
WQ = np.array([[1,0,0,1],[0,1,1,0],[1,0,0,1],[0,1,1,0]], float)  # d_model->H*hd
WK = np.array([[0,1,1,0],[1,0,0,1],[0,1,1,0],[1,0,0,1]], float)
WV = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], float)
print("W_Q (4x4), W_K (4x4), W_V (4x4) map d_model -> n_heads*head_dim=4\n")

Q = H @ WQ                     # (3,4)
K = H @ WK                     # (3,4)  <-- CACHED
V = H @ WV                     # (3,4)  <-- CACHED
print("Q = H@WQ =\n", Q)
print("K = H@WK =\n", K, "   <-- MHA caches this")
print("V = H@WV =\n", V, "   <-- and this")

# split into 2 heads of dim 2
def heads(X): return X.reshape(3, n_heads, head_dim)
Qh, Kh, Vh = heads(Q), heads(K), heads(V)

outs = []
for h in range(n_heads):
    scores = (Qh[q_idx, h] @ Kh[:, h].T) * scale        # (3,)
    a = sm(scores)
    o = a @ Vh[:, h]                                     # (2,)
    print(f"\nhead {h}:  q={Qh[q_idx,h]}  scores={scores}  softmax={a}  out={o}")
    outs.append(o)
concat = np.concatenate(outs)
print("\nconcat heads =", concat)
WO = np.eye(4)
mha_out = concat @ WO
print("MHA block output (after W_O) =", mha_out)
print(f"\nMHA KV cache = K(3x4)+V(3x4) = {K.size + V.size} numbers")

# ======================================================================
# (2) MLA  (same head structure, but factorized + cached latent)
# ======================================================================
print("\n"+"="*70); print("(2) MLA"); print("="*70)
WDKV = np.array([[1,0],[0,1],[1,0],[0,1]], float)        # d_model -> d_c=2   (down)
WUK  = np.array([[0,1,1,0],[1,0,0,1]], float)            # d_c=2 -> H*hd=4    (up, per-head slices)
WUV  = np.array([[1,0,1,0],[0,1,0,1]], float)            # d_c=2 -> H*hd=4
print("W_DKV (4x2) down-proj,  W_UK (2x4) & W_UV (2x4) up-proj\n")

C = H @ WDKV                                             # (3,2)  <-- MLA caches ONLY this
print("latent  C = H@WDKV =\n", C, "   <-- MLA caches ONLY this (2 nums/token)")
print(f"MLA KV cache = C(3x2) = {C.size} numbers   (vs {K.size+V.size} for MHA)")

# --- 2a. MATERIALIZED path: rebuild K,V then do normal MHA ---
Kr = C @ WUK                                             # (3,4)
Vr = C @ WUV
Krh, Vrh = heads(Kr), heads(Vr)
# queries: MLA uses its own WQ (reuse same WQ here for a fair head-to-head)
Qm = H @ WQ; Qmh = heads(Qm)
outs_mat = []
for h in range(n_heads):
    s = (Qmh[q_idx,h] @ Krh[:,h].T)*scale
    outs_mat.append(sm(s) @ Vrh[:,h])
print("\n[materialized]  rebuild K=C@WUK, V=C@WUV, then normal attention")
print("   reconstructed K =\n", Kr)

# --- 2b. ABSORBED path: never rebuild K; fold WUK into the query ---
print("\n[absorbed]  fold WUK into query: score = (q_head) . (WUK_head @ c_i), no K rebuild")
WUK_h = WUK.reshape(n_heads, head_dim, d_c)              # per-head slice (2x2)
outs_abs = []
for h in range(n_heads):
    q_head = Qmh[q_idx, h]                               # (2,)
    q_prime = q_head @ WUK_h[h]                          # (2,) lives in LATENT space d_c
    s = (q_prime @ C.T) * scale                          # dot latent-vs-latent, no K!
    a = sm(s)
    # value side: mix latents first, then one up-proj
    ctx_latent = a @ C                                   # (2,) still in latent space
    o = ctx_latent @ WUV_h if False else (a @ (C @ WUV).reshape(3,n_heads,head_dim)[:,h])
    outs_abs.append(o)
    print(f"   head {h}: q'={q_prime} (latent)  scores={s}  softmax={a}")

print("\nmaterialized out heads:", [np.round(o,3) for o in outs_mat])
print("absorbed    out heads:", [np.round(o,3) for o in outs_abs])
print("=> identical:", np.allclose(np.concatenate(outs_mat), np.concatenate(outs_abs)))
