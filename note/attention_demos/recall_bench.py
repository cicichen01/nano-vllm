import numpy as np
rng = np.random.default_rng(0)
d_k = d_v = 32
def norm(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
def cos_each(Y,V):
    return (Y*V).sum(1)/(np.linalg.norm(Y,axis=1)*np.linalg.norm(V,axis=1)+1e-9)
def cos(Y,V): return cos_each(Y,V).mean()

def build(K,V,mode,beta=1.0):
    S=np.zeros((d_k,d_v))
    if mode=="plain":
        for k,v in zip(K,V): S=S+np.outer(k,v)
    elif mode=="delta":
        for k,v in zip(K,V): S=S+beta*np.outer(k,v-k@S)
    elif mode=="ideal":
        S=np.linalg.pinv(K)@V
    return S

print("Exp 1  mean cosine(recalled, true) over ALL stored items   (d_k=d_v=32)\n")
print(f"{'#items M':>9} {'plain':>7} {'delta':>7} {'ideal':>7}")
for M in [4,8,16,32,48,64]:
    K=norm(rng.standard_normal((M,d_k))); V=rng.standard_normal((M,d_v))
    r={m:cos(K@build(K,V,m),V) for m in ["plain","delta","ideal"]}
    print(f"{M:>9} {r['plain']:>7.3f} {r['delta']:>7.3f} {r['ideal']:>7.3f}")

print("\nExp 1b  RECENCY: overloaded memory M=64; recall of OLDEST-8 vs NEWEST-8 items")
M=64; K=norm(rng.standard_normal((M,d_k))); V=rng.standard_normal((M,d_v))
for mode in ["plain","delta"]:
    ce=cos_each(K@build(K,V,mode),V)
    print(f"  {mode:6s}: oldest-8 = {ce[:8].mean():.3f}   newest-8 = {ce[-8:].mean():.3f}")
print("  (plain ≈ uniform interference; delta sacrifices old for near-perfect RECENT recall)")

print("\nExp 2  OVERWRITE: store 20 facts, revise 6; recall the revised keys")
M,J=20,6
K=norm(rng.standard_normal((M,d_k)))
Vold=rng.standard_normal((M,d_v)); Vnew=rng.standard_normal((J,d_v)); idx=np.arange(J)
sK=np.vstack([K,K[idx]]); sV=np.vstack([Vold,Vnew])
for mode in ["plain","delta"]:
    rec=K[idx]@build(sK,sV,mode)
    print(f"  {mode:6s}: cos(recall,NEW)={cos(rec,Vnew):+.3f}   cos(recall,OLD)={cos(rec,Vold[idx]):+.3f}")
print("  (good = high cos with NEW value, low/neg with OLD value)")
