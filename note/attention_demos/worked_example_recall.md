# Worked example: recall quality — plain linear attention vs DeltaNet

Step-by-step numbers behind `recall_bench.py` / `gdn_demo.py`. Everything here is
hand-followable; the exact values are produced by `worked.py` (small dims) and
`recall_bench.py` (at scale).

Recall the two operations of the fixed-state memory `S` (shape `d_k × d_v`):
- **write** token `(k, v)`:  update `S`
- **read**  query `q`:  `y = qᵀ S`
- plain linear attention:  `S ← S + k vᵀ`                    (just accumulate)
- DeltaNet (β=1):          `S ← S + k (v − kᵀS)ᵀ`  =  `(I − kkᵀ)S + k vᵀ`  (erase old, write new)

---

## Part 1 — interference with overlapping (non-orthogonal) keys

**Setup** (`d_k = d_v = 3`). Three tokens; the third key **overlaps** the first two:

```
k1 = [1, 0, 0]   v1 = [1, 0, 0]
k2 = [0, 1, 0]   v2 = [0, 1, 0]
k3 = [0.6, 0.8, 0]   v3 = [0, 0, 1]      k3·k1 = 0.6,  k3·k2 = 0.8   (NOT orthogonal)
```
Goal: after all three writes, reading each key should return its own value.

### Plain linear attention — build `S = Σ kᵢvᵢᵀ`

```
token 1:  S += outer(k1,v1)          token 2:  S += outer(k2,v2)
S = [[1 0 0]                          S = [[1 0 0]
     [0 0 0]                               [0 1 0]
     [0 0 0]]                              [0 0 0]]

token 3:  outer(k3,v3) = [[0 0 0.6]        final S = [[1 0 0.6]
                          [0 0 0.8]                   [0 1 0.8]
                          [0 0 0  ]]                  [0 0 0  ]]
```
`outer(k3,v3)` writes `v3=[0,0,1]` into the **rows of k3** (rows 0,1 weighted 0.6, 0.8),
so it spills into the `k1` and `k2` rows.

**Read (each `y = kᵢ @ S`):**
```
read k1 = [1, 0, 0]·S = [1, 0, 0.6]      want [1,0,0]   ✗ 0.6 leaked from k3
read k2 = [0, 1, 0]·S = [0, 1, 0.8]      want [0,1,0]   ✗ 0.8 leaked from k3
read k3 = [0.6,0.8,0]·S = [0.6, 0.8, 1]  want [0,0,1]   ✗ 0.6,0.8 leaked from k1,k2
```
Decomposing the last one shows the interference explicitly:
```
read k3 = (k3·k1)v1 + (k3·k2)v2 + (k3·k3)v3
        = 0.6·[1,0,0] + 0.8·[0,1,0] + 1·[0,0,1] = [0.6, 0.8, 1.0]
          └ junk from k1 ┘ └ junk from k2 ┘ └ the real answer ┘
```

### DeltaNet — build `S = S + kᵢ(vᵢ − kᵢᵀS)ᵀ`

```
token 1:  v_old = k1·S = [0,0,0]   err = v1 − v_old = [1,0,0]
          S = [[1 0 0],[0 0 0],[0 0 0]]

token 2:  v_old = k2·S = [0,0,0]   err = [0,1,0]
          S = [[1 0 0],[0 1 0],[0 0 0]]

token 3:  v_old = k3·S = [0.6, 0.8, 0]         ← k3 ALREADY reads junk from k1,k2!
          err   = v3 − v_old = [0,0,1] − [0.6,0.8,0] = [−0.6, −0.8, 1]
          outer(k3, err) = [[−0.36 −0.48  0.6]
                            [−0.48 −0.64  0.8]
                            [ 0     0     0 ]]
          final S = [[ 0.64 −0.48  0.6]
                     [−0.48  0.36  0.8]
                     [ 0     0     0 ]]
```
The key step is **token 3**: before writing, DeltaNet *reads what's already there for k3*
(`v_old = [0.6,0.8,0]` — the interference!), computes the **error** `err = v3 − v_old`,
and writes that. The negative entries `−0.36, −0.48, …` are DeltaNet **subtracting out**
the contamination that k1,k2 would have caused.

**Read:**
```
read k3 = [0.6,0.8,0]·S = [0, 0, 1.0]        want [0,0,1]   ✓ EXACT — interference cancelled
read k1 = [1,0,0]·S     = [0.64, −0.48, 0.6] want [1,0,0]   ~ degraded (recency cost)
read k2 = [0,1,0]·S     = [−0.48, 0.36, 0.8] want [0,1,0]   ~ degraded
```

### Side-by-side (reading k3, the just-written key)
```
plain :  read k3 = [0.6, 0.8, 1.0]   ← contaminated by k1, k2
delta :  read k3 = [0.0, 0.0, 1.0]   ← clean
```

**What it teaches:** DeltaNet's error-correcting write gives the **most-recently-written
key clean recall** even when keys overlap, by subtracting the interference at write time.
The honest cost: it does so partly at the expense of the **earlier** keys (k1, k2 degrade)
— a recency bias (see Part 3).

---

## Part 2 — overwrite (a key is revised)

**Setup** (`d_k = d_v = 2`). Token 3 reuses token 1's key with a new value:
```
token1: k=[1,0] v=[1,0]      token2: k=[0,1] v=[0,1]      token3: k=[1,0] v=[0,1]
```
Desired (overwrite/latest-wins): read A=[1,0] → [0,1].

```
                         final S            read A=[1,0]    verdict
plain  Σkvᵀ :  [[1,1],[0,1]]                 [1, 1]         ✗ old+new summed
delta      :  [[0,1],[0,1]]                 [0, 1]         ✓ old value erased, new kept
```
Plain sums `v_old=[1,0]` and `v_new=[0,1]` → `[1,1]`. DeltaNet's `(I−kkᵀ)` term erases the
`[1,0]` first, so only `[0,1]` remains.

---

## Part 3 — at scale (`d_k = d_v = 32`, random unit keys)

Mean cosine(recalled, true) over all stored items (`recall_bench.py`):

```
 #items M   plain   delta   ideal K⁺V
        4    0.935   0.951   1.000     lightly loaded: delta > plain
        8    0.894   0.935   1.000
       16    0.842   0.842   1.000
       32    0.704   0.682   1.000     at capacity: ideal still perfect
       48    0.621   0.520   0.803     overloaded: all degrade
       64    0.551   0.419   0.689
```

Recency (overloaded, M=64): recall of oldest-8 vs newest-8 items:
```
plain :  oldest-8 = 0.565   newest-8 = 0.587    (≈ uniform smear)
delta :  oldest-8 = 0.151   newest-8 = 0.906    (recent near-perfect; old sacrificed)
```

Overwrite (store 20, revise 6; recall revised keys):
```
plain :  cos(recall, NEW) = +0.557   cos(recall, OLD) = +0.620   (keeps BOTH)
delta :  cos(recall, NEW) = +0.893   cos(recall, OLD) = −0.016   (clean overwrite)
```

---

## Part 4 — same data through SOFTMAX (original) attention

Softmax attention does **not** compress into a fixed `S`. It keeps **all** tokens (the KV
cache) and, per query, computes `softmax(q·kᵢ over all i)` then a weighted sum of values.
So recall is `y = softmax(scale · K·q) · V`. Run: `recall_softmax.py`.

### Exp 1 — recall vs #items (interference / capacity)
```
   M   plain   delta   softmax(1/√d)   softmax(sharp)
   4   0.935   0.951        0.539          1.000
   8   0.894   0.935        0.375          1.000
  16   0.842   0.842        0.286          1.000
  32   0.704   0.682        0.205          1.000     ← plain/delta at d_k ceiling
  48   0.621   0.520        0.163          1.000
  64   0.551   0.419        0.170          1.000
 128   0.449   0.261        0.131          1.000     ← M=128 » d_k=32, softmax STILL 1.000
```
**This is how softmax avoids the recall issue:** with adequately separable keys
(`sharp` column), it recalls **perfectly at every M — no `d_k` ceiling** — because it never
merges tokens into a bounded state. Each query re-selects among all retained tokens.

Two honest caveats the table shows:
- `softmax(1/√d)` is *worse* than plain here — with tiny-margin **random** keys, the
  `1/√d` scale is too soft to pick the right token. Real models don't fail this way because
  **training makes keys separable**, so the standard scale is sharp *enough* (the `sharp`
  column simulates that). So softmax's win requires distinguishable keys, not magic.
- Softmax pays for it in **memory**: it stores all `M` tokens (`O(M)` KV cache) instead of a
  fixed `O(d_k²)` state. That is exactly the cache linear attention/GDN is trying to avoid.

### Exp 2 — overwrite (identical keys): softmax does NOT solve it either
```
  plain   : cos(recall,NEW)=+0.556  cos(recall,OLD)=+0.655
  delta   : cos(recall,NEW)=+0.942  cos(recall,OLD)=+0.039   ← only this erases
  softmax : cos(recall,NEW)=+0.680  cos(recall,OLD)=+0.704   ← blends, like plain
```
With **truly identical** keys, both revised tokens get the same score, so softmax weights
them equally and **blends** old+new — no better than plain, and worse than delta's explicit
erase. Real LMs sidestep this because position (RoPE) / context make the two keys **differ**,
and softmax — retaining both tokens — can then learn to attend to the latest. So softmax's
"overwrite" is *learned selection over distinct retained keys*, not an intrinsic erase.

### The clean split
```
capacity / interference (distinct keys):  softmax has NO issue (retention, no d_k ceiling)
                                           — at O(M) memory cost.
overwrite (identical keys):                softmax ALSO blends; only delta erases.
                                           real LMs make keys distinct (position) so softmax selects.
```

---

## Takeaways

1. **Overwrite is DeltaNet's unambiguous win** (Part 1 read-k3, Part 2, Part 3 overwrite):
   plain sums old+new; DeltaNet subtracts the old at write time and returns the latest.
2. **Recency is a feature** (Part 1 k1/k2 degrade, Part 3 recency): DeltaNet concentrates
   recall on recent items instead of smearing interference uniformly — what LMs want, and
   what the gate `α` reinforces with explicit decay.
3. **Capacity is capped at `d_k` for both** (Part 3 `ideal` = 1.000 only up to M=32=d_k):
   a fixed state cannot exceed `d_k` clean associations. DeltaNet uses the budget better
   (correlation `Σkvᵀ` → regression `K⁺V`), it does not raise the ceiling.
4. **Distinct/orthogonal keys already recall fine in plain** — interference needs
   *overlapping* keys (Part 1 uses k3·k1=0.6, k3·k2=0.8). Random keys in high-dim are only
   near-orthogonal, so interference grows with the number of items.
5. **Caveat:** this uses random keys and fixed β=1, understating real GDN (learned keys are
   more orthogonal; β and the gate α are data-dependent), so the practical gap favors GDN
   more than shown here.
