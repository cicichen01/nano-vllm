import numpy as np

d_model, n_heads, head_dim, d_c = 4096, 32, 128, 512
# note: n_heads*head_dim = 4096 = d_model

print("KEY FACT: W_UK is PER-HEAD (that's what makes MLA differ from MQA).")
print("The shared latent c is d_c=512, but EACH head decompresses it with its")
print("OWN slice W_UK^(h) (128x512). If W_UK were shared -> that would be MQA.\n")

# full matrices (all heads)
WQ_all  = (n_heads*head_dim, d_model)   # 4096 x 4096
WUK_all = (n_heads*head_dim, d_c)       # 4096 x 512   <- reshaped to 32 x (128 x 512)
print("all-heads factored:")
print(f"  W_Q  {WQ_all}  = {np.prod(WQ_all):,}")
print(f"  W_UK {WUK_all} = {np.prod(WUK_all):,}")
fact_all = np.prod(WQ_all)+np.prod(WUK_all)
print(f"  total factored (all heads) = {fact_all:,}")

# folded absorbed query, PER HEAD:  W_UK^(h)^T @ W_Q^(h)  -> (d_c x d_model) = 512x4096
WQp_perhead = (d_c, d_model)            # 512 x 4096  PER HEAD
print(f"\nfolded absorbed query W_Q'^(h) = W_UK^(h)^T @ W_Q^(h): {WQp_perhead} PER HEAD")
print(f"  per head       = {np.prod(WQp_perhead):,}")
print(f"  x {n_heads} heads = {n_heads*np.prod(WQp_perhead):,}   (all heads)")

print(f"\ncompare (all heads):  factored {fact_all:,}  vs  folded "
      f"{n_heads*np.prod(WQp_perhead):,}  = {n_heads*np.prod(WQp_perhead)/fact_all:.1f}x")

# sanity: per-head factored vs per-head folded
ph_fact = head_dim*d_model + head_dim*d_c
ph_fold = d_c*d_model
print(f"\nper head:  factored {ph_fact:,}  vs  folded {ph_fold:,}  = {ph_fold/ph_fact:.1f}x")
print("\n=> both are PER HEAD (there are n_heads of each). Ratio 3.6x holds either way.")
