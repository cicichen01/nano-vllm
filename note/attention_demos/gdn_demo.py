import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# State S is (d_k x d_v) associative memory.
#   write (k,v):  contributes outer(k, v)      -> "store value v under key k"
#   read  q:      y = qᵀ S = Σ_i (q·k_i) v_i   -> linear attention, NO softmax
#
# Sequence of 3 writes; the 3rd UPDATES key A (should OVERWRITE its old value).
A = np.array([1., 0.]); B = np.array([0., 1.])
seq = [("store A→[1,0]", A, np.array([1., 0.])),
       ("store B→[0,1]", B, np.array([0., 1.])),
       ("UPDATE A→[0,1]", A, np.array([0., 1.]))]   # overwrite A

I = np.eye(2)
def run(mode, beta=1.0, alpha=1.0):
    S = np.zeros((2, 2))
    print(f"\n===== {mode}  (beta={beta}, alpha={alpha}) =====")
    for name, k, v in seq:
        if mode == "linear attention":
            S = S + np.outer(k, v)                                  # just ADD, never forget
        else:  # delta / gated-delta
            S = alpha * ((I - beta*np.outer(k, k)) @ S) + beta*np.outer(k, v)
        print(f"  after {name:15s}  S =\n{S}")
    print(f"  READ A -> {A @ S}      READ B -> {B @ S}")
    return S

run("linear attention")                       # interference: A never forgets old value
run("DeltaNet",        beta=1.0, alpha=1.0)    # delta rule: clean overwrite
run("Gated DeltaNet",  beta=1.0, alpha=0.7)    # + forget gate: overwrite AND old B fades

print("\nGoal after the 3 writes:  READ A should give [0,1] (the updated value),")
print("                          READ B should give [0,1] (untouched).")
