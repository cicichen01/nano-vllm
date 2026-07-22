import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# 3-dim keys/values; k3 OVERLAPS k1 and k2 (non-orthogonal) -> interference test
k1=np.array([1.,0,0]); v1=np.array([1.,0,0])
k2=np.array([0.,1,0]); v2=np.array([0.,1,0])
k3=np.array([0.6,0.8,0]); v3=np.array([0.,0,1])
K=[k1,k2,k3]; V=[v1,v2,v3]
print("k3·k1 =", k3@k1, "  k3·k2 =", k3@k2, "  (k3 overlaps both)\n")

print("###### PLAIN: S = Σ outer(k,v) ######")
S=np.zeros((3,3))
for i,(k,v) in enumerate(zip(K,V),1):
    o=np.outer(k,v); S=S+o
    print(f"token {i}: outer(k{i},v{i})=\n{o}\nrunning S=\n{S}\n")
Sp=S.copy()
for i,k in enumerate(K,1):
    print(f"read k{i}: y = k{i}@S = {k@S}   (want v{i}={V[i-1]})")

print("\n###### DELTA: S = S + outer(k, v - k@S) ######")
S=np.zeros((3,3))
for i,(k,v) in enumerate(zip(K,V),1):
    vold=k@S; err=v-vold; o=np.outer(k,err); S=S+o
    print(f"token {i}: v_old=k{i}@S={vold}  err=v{i}-v_old={err}\n outer(k{i},err)=\n{o}\n running S=\n{S}\n")
for i,k in enumerate(K,1):
    print(f"read k{i}: y = k{i}@S = {k@S}   (want v{i}={V[i-1]})")
