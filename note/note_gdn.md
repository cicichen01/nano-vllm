# Gated DeltaNet (GDN) — Notes

Built from first principles. GDN is a **linear-attention** sequence mixer: no KV cache,
a fixed-size recurrent state, O(N) compute / O(1) memory. This note traces *why* it's built
the way it is. Companion material in `attention_demos/` (scripts, `state_growth.md`,
`weight_formulas.md`, `worked_example_recall.md`) and figures in `../h100_setup/`.

---

## 1. Motivation chain

```
softmax attention   great recall, but O(N) KV cache (keep every k,v) + O(N²) compute
   │  drop softmax → can re-associate (qKᵀ)V = q(KᵀV); fold history into a fixed state S
   ▼
plain linear attn   S = Σ kᵢvᵢᵀ,  y = qᵀS.  O(1) memory ✓  BUT poor recall ✗
   │  fix recall without growing memory → treat writing as ONLINE LEARNING of a k→v map
   ▼
DeltaNet            error-correcting write (delta rule): S ← (I−βkkᵀ)S + βkvᵀ
   │  add controllable forgetting / recency / capacity management
   ▼
Gated DeltaNet      + data-dependent forget gate α:  S ← α(I−βkkᵀ)S + βkvᵀ
```

**Why recall matters:** copying, in-context learning, coreference, long-context lookup are
all *retrieval*. Empirically (Zoology / MQAR) the recall gap explains most of the real
LM-quality gap between subquadratic models and Transformers. So closing it is the point.

---

## 2. The state is an associative memory; readout is attention

`S` (shape `d_k × d_v`, here per head; `d_k,d_v` ≙ `head_dim`) is a **key→value lookup table**.
- write `(k,v)`: update `S`.  read `q`:  `y = qᵀS = Σᵢ (q·kᵢ) vᵢ`.
- `y = qᵀS` **is** attention, just with the token-sum folded into `S` (weights `q·kᵢ` live inside).
- **Recall** = query with a stored key `kⱼ`, expect `vⱼ` back. Clean recall needs the weight
  distribution to concentrate on `j`, which needs **distinguishable keys** (`kⱼ·kⱼ > kⱼ·kᵢ`).

State size is **quadratic in head_dim** (`d_k·d_v`, e.g. 128×128=16384) but **fixed** — vs
MHA's `2·d_model` *per token* (grows). Crossover ~64 tokens; beyond that GDN's flat state wins.

---

## 3. Plain linear attention and its two failures

`S = Σᵢ kᵢ⊗vᵢ` (Hebbian/correlation memory). Problems:
1. **Interference** — `kⱼᵀS = vⱼ + Σ_{i≠j}(kⱼ·kᵢ)vᵢ`; overlapping keys contaminate recall
   (clean only if keys orthonormal).
2. **No overwrite** — a repeated key sums its values (`[1,0]+[0,1]=[1,1]` instead of latest `[0,1]`).
3. **Capacity** — `rank(S) ≤ d_k`; grows 1/token, saturates at `d_k`. Beyond `d_k` distinct
   keys, associations *must* collide.
4. **No position** — `S=Σkvᵀ` is a **commutative sum** → final `S` is **order-invariant**
   (permutation-invariant "bag of tokens"). Needs explicit position encoding.

---

## 4. DeltaNet: writing = online error-correcting learning (the "minus")

Goal: make `S` satisfy `kᵢᵀS ≈ vᵢ` (a regression, not blind accumulation). Online rule:
```
predictionₜ = kₜᵀ S_{t-1}
errorₜ      = vₜ − predictionₜ            ← "write what's MISSING, not the whole value"
S_t         = S_{t-1} + kₜ ⊗ errorₜ  =  (I − kₜkₜᵀ)S_{t-1} + kₜvₜᵀ   (β=1)
```
This is the classic **delta / Widrow–Hoff / LMS rule** = one step of gradient descent on
`½‖kₜᵀS − vₜ‖²`. The subtraction cures the failures:
- **Write-time exactness:** right after writing `i`, `kᵢᵀSᵢ = vᵢ` **exactly** (error zeroed).
- **Overwrite:** repeated key → `error = vₜ − old` replaces old with new.
- **Decorrelation:** drives `S` toward the regression/pseudoinverse `K⁺V` instead of
  correlation `KᵀV`, cancelling cross-key interference.

**Equivalent forms** (all identical, verified in `verify.py`):
```
matrix:      S = (I − kkᵀ)S + kvᵀ
outer/error: S = Σᵢ kᵢ⊗eᵢ ,  eᵢ = vᵢ − Σ_{l<i}(kᵢ·kₗ)eₗ
```

### The per-token weights (what each mechanism does to raw similarities)
```
linear :  wⱼ = q·kⱼ
softmax:  wⱼ = e^{q·kⱼ/√d} / Σₗ e^{q·kₗ/√d}                       (exponential margin)
delta  :  w  = [q·k₁ … q·k_M]·(I+L)⁻¹ ,  Lⱼᵢ=(kⱼ·kᵢ if i<j else 0)  (decorrelated)
          per element: wⱼ = q·kⱼ + Σ_{l>j}(q·kₗ)[(I+L)⁻¹]ₗⱼ
```
`(I+L)⁻¹ = I − L + L² − …` (finite; L nilpotent). If keys orthogonal, `L=0`, `(I+L)⁻¹=I`,
and **DeltaNet = linear attention.** So `L` (key overlaps) is the entire difference.
Figures: `attention_weights.png`, `attention_weight_formulas.png`, `deltanet_inverse_matrix.png`.

**Softmax vs delta — two routes to clean recall:**
- softmax concentrates by **exponential margin** (needs target score ≫ others; small margins → blends).
- delta cleans by **exact subtraction** (margin-independent; can beat raw softmax on correlated keys).

---

## 5. Gated DeltaNet: add a forget gate

```
S_t = α_t · (I − β_t kₜkₜᵀ) S_{t-1} + β_t kₜvₜᵀ
      └ α: forget ┘└─ β: overwrite at key kₜ ─┘
```
Two knobs, both **data-dependent** (per token, computed from the token's hidden state `h_t`
via learned projections + sigmoid — NOT global constants; the *projections* are fixed):
- **β_t (write strength, per key):** how hard to overwrite the value at `kₜ`. β=1 full
  replace at that key; β<1 blend old+new *at that key*; β=0 ignore token. Targeted.
- **α_t (forget, global):** multiplies the *whole* `S` → shrinks **all** stored associations
  uniformly. Manages capacity + recency; without it (DeltaNet, α=1) memory never fades.

DeltaNet = GDN with α=1. Plain linear = also drop the `(I−βkkᵀ)` erase term.

**Granularity of α:** in vanilla GDN it's a **scalar per head** (following Mamba2) — uniform
across keys/dims in a head, for efficiency (keeps the chunked-parallel kernel clean). Finer
per-channel gates exist (GLA = per-dimension; **Kimi's KDA** = finer channel-wise gating).

---

## 6. Position & order

- **Plain linear:** order-invariant sum → **no position encoded** (verified: reordering tokens
  gives identical final `S`). Needs explicit PE.
- **DeltaNet:** the non-commutative recurrence makes final `S` **order-dependent** → implicit
  recency/order (later overlapping writes overwrite earlier ones). Verified: `S` changes with order.
- **GDN gate / RetNet decay:** explicit recency decay strengthens this.
- **Gates are input-only** (`α_t=f(h_t)`, not of `S_{t-1}`) — deliberate, to keep the recurrence
  parallelizable. So a gate can't compare current-to-previous itself; its context-sensitivity is
  **borrowed** — earlier layers contextualize `h_t`, the gate reads it. Shallow layers ≈
  token/position signals; deep layers ≈ context-based.

**Real hybrid models** (mostly GDN + a few full-attention layers): the GDN layers use **no
explicit PE**; the attention layers vary — **Qwen3-Next** uses partial RoPE on attention;
**Kimi Linear** uses **NoPE** MLA (KDA supplies position); **Jamba** (Mamba hybrid) uses NoPE
throughout. So NoPE attention is real — it relies on the linear layers for order.

---

## 7. Recall: capacity ceiling and the recency trade

- **Capacity separation (theory):** fixed state holds `~d_k` associations; softmax keeps all
  `N` → there's a recall task softmax solves that any fixed state can't.
- **Empirical (`recall_proof.py`):** exact-retrieval accuracy — softmax = 1.00 for N=8…256
  (distinct keys, no ceiling); linear/delta perfect until `N≈d_k`, then fall off.
- **Recency trade:** the bare delta rule (β=1, no gate) gives clean recall of *recent/just-
  written/overwritten* keys but **sacrifices old ones** — on a *uniform* all-items probe it can
  even trail plain linear attention (`recall_bench.py`). Real GDN's recall win comes from
  **learned (more-orthogonal) keys + data-dependent β + the gate α + training**, on the
  recency/overwrite-flavored recall that real LMs need — not on uniform synthetic recall.
- Neither linear nor delta escapes the `d_k` ceiling → hybrids keep a few softmax/MLA layers
  for *uncapped* recall.

---

## 8. Where GDN sits

```
                cache/token     compute    recall              position
MHA (softmax)   2·d_model (grows) O(N²)     exact, uncapped     RoPE
MLA             ~d_c   (grows)     O(N²)     exact (low-rank KV) RoPE (decoupled)
GDN / linear    NONE (fixed S)    O(N)      good, capped ~d_k   implicit (recurrence+gate)
```
GDN and MLA are opposite bets (recurrent state vs compressed exact-attention cache) and
**compose** in hybrids (e.g. Kimi Linear: KDA + MLA).

---

## 9. Papers (lineage)

- Linear attention origin — Katharopoulos et al., *Transformers are RNNs*, ICML 2020 (2006.16236).
- Delta rule in linear attn — Schlag, Irie, Schmidhuber, *Linear Transformers are Secretly Fast
  Weight Programmers*, ICML 2021 (2102.11174).
- DeltaNet parallel training — Yang, Wang, Zhang, Shen, Kim, NeurIPS 2024 (2406.06484).
- **Gated DeltaNet** — Yang, Kautz, Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with
  Delta Rule*, ICLR 2025 (2412.06464).

---

## 10. Demos & figures

- `attention_demos/gdn_demo.py` — associative memory: linear interference vs delta overwrite vs gate.
- `attention_demos/gdn_vs_mha.py` — softmax readout vs `y=qᵀS`.
- `attention_demos/recall_bench.py`, `recall_softmax.py`, `recall_proof.py` — recall vs capacity/recency/overwrite; +softmax.
- `attention_demos/order_dep.py` — order-(in)dependence of `S`.
- `attention_demos/state_growth.md`, `weight_formulas.md`, `worked_example_recall.md` — step-by-step.
- figures: `../h100_setup/gdn_state_update.png`, `attention_weights.png`,
  `attention_weight_formulas.png`, `deltanet_inverse_matrix.png`, `attn_kvcache_growth.png`, `hybrid_stack.png`.
