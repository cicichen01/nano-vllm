import numpy as np
rng=np.random.default_rng(5); np.set_printoptions(precision=4,suppress=True)
toks=[(rng.standard_normal(3),rng.standard_normal(3)) for _ in range(3)]  # (k,v) x3
I=np.eye(3)
def build(order, mode):
    S=np.zeros((3,3))
    for idx in order:
        k,v=toks[idx]
        S = S+np.outer(k,v) if mode=="plain" else (I-np.outer(k,k))@S+np.outer(k,v)
    return S

for mode in ["plain","delta"]:
    S123=build([0,1,2],mode); S321=build([2,1,0],mode); S213=build([1,0,2],mode)
    print(f"\n=== {mode} : final S for different token ORDERS ===")
    print("order 1,2,3 vs 3,2,1  identical? ", np.allclose(S123,S321))
    print("order 1,2,3 vs 2,1,3  identical? ", np.allclose(S123,S213))
    if not np.allclose(S123,S321):
        print(" S(1,2,3)=\n",S123,"\n S(3,2,1)=\n",S321)
