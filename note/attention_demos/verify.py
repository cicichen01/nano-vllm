import numpy as np
rng=np.random.default_rng(3); np.set_printoptions(precision=5,suppress=True)
k1,k2,k3=[rng.standard_normal(3) for _ in range(3)]
v1,v2,v3=[rng.standard_normal(3) for _ in range(3)]
I=np.eye(3)

# (A) matrix update:  S_t = (I - k kᵀ) S_{t-1} + k vᵀ   (my convention: S is d_k×d_v)
S=np.zeros((3,3))
for k,v in [(k1,v1),(k2,v2),(k3,v3)]:
    S=(I-np.outer(k,k))@S + np.outer(k,v)

# (B) e-form:  S = Σ k_i ⊗ e_i
a=k2@k1; b=k3@k1; c=k3@k2
e1=v1; e2=v2-a*v1; e3=v3-b*v1-c*e2
Se=np.outer(k1,e1)+np.outer(k2,e2)+np.outer(k3,e3)

# (C) fully expanded, k&v only (no e):  per entry
def S_kv(r,cc):
    return (k1[r]*v1[cc]+k2[r]*v2[cc]+k3[r]*v3[cc]
            - a*k2[r]*v1[cc]
            - c*k3[r]*v2[cc]
            - (b-c*a)*k3[r]*v1[cc])
Skv=np.array([[S_kv(r,cc) for cc in range(3)] for r in range(3)])

print("(A) matrix (I-kkᵀ)S+kvᵀ  ==  (B) e-form :", np.allclose(S,Se))
print("(A)                       ==  (C) k,v-only:", np.allclose(S,Skv))
