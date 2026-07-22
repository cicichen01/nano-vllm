import numpy as np
rng=np.random.default_rng(0)
d_k=d_v=32                                  # FIXED state = 32x32 for linear/delta; softmax keeps all N
def norm(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)

def acc(retrieved, V):                       # exact-retrieval accuracy: nearest stored value is the right one
    d=((retrieved[:,None,:]-V[None,:,:])**2).sum(-1)   # (N,N) distances to every stored value
    return (d.argmin(1)==np.arange(len(V))).mean()

def state(K,V,mode):
    S=np.zeros((d_k,d_v))
    for k,v in zip(K,V):
        S=S+(np.outer(k,v) if mode=="plain" else np.outer(k,v-k@S))
    return K@S
def soft(K,V,scale=8.0):
    s=(K@K.T)*scale; s-=s.max(1,keepdims=True); a=np.exp(s); a/=a.sum(1,keepdims=True); return a@V

print("EXACT-RETRIEVAL ACCURACY (query each stored key, did we get its value back?)")
print(f"fixed state d_k=d_v=32;  softmax keeps all N tokens\n")
print(f"{'N pairs':>8} {'linear':>8} {'delta':>8} {'softmax':>9}")
for N in [8,16,32,64,128,256]:
    K=norm(rng.standard_normal((N,d_k))); V=rng.standard_normal((N,d_v))
    print(f"{N:>8} {acc(state(K,V,'plain'),V):>8.2f} {acc(state(K,V,'delta'),V):>8.2f} {acc(soft(K,V),V):>9.2f}")
