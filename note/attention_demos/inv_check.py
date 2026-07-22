import numpy as np
rng=np.random.default_rng(1); np.set_printoptions(precision=4,suppress=True)
K=rng.standard_normal((4,4))
def a(i,j): return K[i-1]@K[j-1]      # k_i · k_j  (1-indexed)
L=np.tril(K@K.T,-1); B=np.linalg.inv(np.eye(4)+L)
# my symbolic (I+L)^{-1}, unit lower-tri, entries in key dot-products:
S=np.eye(4)
S[1,0]=-a(2,1)
S[2,0]=-a(3,1)+a(3,2)*a(2,1);           S[2,1]=-a(3,2)
S[3,0]=-a(4,1)+a(4,2)*a(2,1)+a(4,3)*a(3,1)-a(4,3)*a(3,2)*a(2,1)
S[3,1]=-a(4,2)+a(4,3)*a(3,2);           S[3,2]=-a(4,3)
print("numeric (I+L)^-1 == symbolic formula:", np.allclose(B,S))
