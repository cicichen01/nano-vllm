import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# 2 context tokens, d_model=4.  Values chosen to expose the difference.
V = np.array([[1.,1., 1.,1.],     # token 0
              [0.,0., 0.,0.]])    # token 1
print("V (2 tokens x d_model=4):\n", V)
TARGET = np.array([1.,1., 0.,0.])
print("\nGOAL for the query's output: [1,1, 0,0]")
print("  = 'take features 0-1 from token 0, features 2-3 from token 1'\n")

# ---------- SINGLE HEAD (head_dim = 4) ----------
print("="*58); print("SINGLE HEAD: ONE softmax 'a' shared by ALL 4 dims"); print("="*58)
print("output = a*V[0] + (1-a)*V[1] = [a,a,a,a]  for ANY weight a")
for a in [1.0, 0.5, 0.0, 0.7]:
    out = a*V[0] + (1-a)*V[1]
    print(f"  a={a:.1f} -> {out}")
# best possible single-head fit to TARGET (search a in [0,1])
grid=np.linspace(0,1,1001)
errs=[np.linalg.norm((a*V[0]+(1-a)*V[1])-TARGET) for a in grid]
ba=grid[int(np.argmin(errs))]
print(f"\nBEST single-head fit: a={ba:.3f} -> {ba*V[0]+(1-ba)*V[1]}")
print(f"  error vs target = {min(errs):.3f}   (CANNOT reach [1,1,0,0])")

# ---------- MHA (2 heads, head_dim = 2) ----------
print("\n"+"="*58); print("MHA: TWO independent softmaxes (a for head0, b for head1)"); print("="*58)
Vh = V.reshape(2, 2, 2)                     # (token, head, head_dim)
a, b = 1.0, 0.0                             # head0 -> token0, head1 -> token1
out0 = a*Vh[0,0] + (1-a)*Vh[1,0]            # [1,1]
out1 = b*Vh[0,1] + (1-b)*Vh[1,1]            # [0,0]
out = np.concatenate([out0, out1])
print(f"head0 weight a={a} -> {out0}   (features 0-1 from token 0)")
print(f"head1 weight b={b} -> {out1}   (features 2-3 from token 1)")
print(f"concat -> {out}")
print(f"  error vs target = {np.linalg.norm(out-TARGET):.3f}   (EXACT)")

print("\nTakeaway: single head ties all dims to ONE distribution -> reachable")
print("outputs are only [x,x,x,x]. MHA's two distributions reach [a,a,b,b].")
