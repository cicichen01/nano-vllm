import numpy as np
rng = np.random.default_rng(0)
d_k = d_v = 32
def norm(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
def cos(Y,V): return ((Y*V).sum(1)/(np.linalg.norm(Y,axis=1)*np.linalg.norm(V,axis=1)+1e-9)).mean()

def state_recall(K,V,mode,beta=1.0):          # linear attn: compress to fixed S
    S=np.zeros((d_k,d_v))
    for k,v in zip(K,V):
        S = S + (np.outer(k,v) if mode=="plain" else beta*np.outer(k, v-k@S))
    return K @ S

def softmax_recall(Kall,Vall,Q,scale):        # softmax attn: keep ALL tokens, select
    sc=(Q@Kall.T)*scale; sc-=sc.max(1,keepdims=True)
    a=np.exp(sc); a/=a.sum(1,keepdims=True)
    return a @ Vall

print("Exp 1  recall vs #items M.  softmax KEEPS ALL M tokens (KV cache, O(M) mem);")
print("       linear compresses to fixed S (O(d_k²)).  d_k=d_v=32\n")
print(f"{'M':>4} {'plain':>7} {'delta':>7} {'softmax(1/√d)':>14} {'softmax(sharp)':>15}")
for M in [4,8,16,32,48,64,128]:
    K=norm(rng.standard_normal((M,d_k))); V=rng.standard_normal((M,d_v))
    row=(cos(state_recall(K,V,"plain"),V), cos(state_recall(K,V,"delta"),V),
         cos(softmax_recall(K,V,K,1/np.sqrt(d_k)),V), cos(softmax_recall(K,V,K,8.0),V))
    print(f"{M:>4} {row[0]:>7.3f} {row[1]:>7.3f} {row[2]:>14.3f} {row[3]:>15.3f}")
print("  (sharp = scale 8 ~ what TRAINING achieves: separable keys; note NO d_k ceiling)")

print("\nExp 2  OVERWRITE (store 20, revise 6 with the SAME key, new value):")
M,J=20,6
K=norm(rng.standard_normal((M,d_k))); Vold=rng.standard_normal((M,d_v)); Vnew=rng.standard_normal((J,d_v)); idx=np.arange(J)
sK=np.vstack([K,K[idx]]); sV=np.vstack([Vold,Vnew])
for name,rec in [("plain",  state_recall(sK,sV,"plain")[idx] if False else K[idx]@ (lambda:
                    __import__('numpy').add.reduce([np.outer(k,v) for k,v in zip(sK,sV)]))()),
                 ]:
    pass
# recompute cleanly
def build(mode):
    S=np.zeros((d_k,d_v))
    for k,v in zip(sK,sV): S=S+(np.outer(k,v) if mode=="plain" else np.outer(k,v-k@S))
    return K[idx]@S
r_plain=build("plain"); r_delta=build("delta")
r_soft =softmax_recall(sK,sV,K[idx],8.0)
for name,r in [("plain",r_plain),("delta",r_delta),("softmax",r_soft)]:
    print(f"  {name:8s}: cos(recall,NEW)={cos(r,Vnew):+.3f}  cos(recall,OLD)={cos(r,Vold[idx]):+.3f}")
print("  (identical keys -> softmax BLENDS old&new too; only delta erases. Real LMs avoid")
print("   this because position/context make the two keys differ, letting softmax select.)")
