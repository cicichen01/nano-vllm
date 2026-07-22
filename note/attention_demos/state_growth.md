# How the state S grows, token by token (plain linear attention vs DeltaNet)

State `S` is `d_k × d_v` (here 3×3). Notation: `k_{i,r}` = token *i*, key-dim *r*;
`v_{i,c}` = token *i*, value-dim *c*. `S[r][c]` = row *r* (key dim), col *c* (value dim).
Readout for a query: `y = qᵀS`. Verified numerically by `verify.py`.

Key overlaps (scalars) used below:
```
(k₂·k₁) = k21·k11 + k22·k12 + k23·k13
(k₃·k₁) = k31·k11 + k32·k12 + k33·k13
(k₃·k₂) = k31·k21 + k32·k22 + k33·k23
```

---

## Plain linear attention:  S = Σᵢ kᵢ ⊗ vᵢ   (each token adds one outer product)

**1 token**
```
      ⎡ k11·v11   k11·v12   k11·v13 ⎤
S  =  ⎢ k12·v11   k12·v12   k12·v13 ⎥          (rank 1)
      ⎣ k13·v11   k13·v12   k13·v13 ⎦
```
**2 tokens** (`S += k₂ ⊗ v₂`)
```
      ⎡ k11·v11+k21·v21   k11·v12+k21·v22   k11·v13+k21·v23 ⎤
S  =  ⎢ k12·v11+k22·v21   k12·v12+k22·v22   k12·v13+k22·v23 ⎥   (rank ≤ 2)
      ⎣ k13·v11+k23·v21   k13·v12+k23·v22   k13·v13+k23·v23 ⎦
```
**3 tokens** (`S += k₃ ⊗ v₃`): general entry `S[r][c] = Σᵢ k_{i,r} v_{i,c}` = `KᵀV`. (rank ≤ 3 = d_k, full)

Fixed size (3×3) forever; each token adds one rank-1 term; rank climbs 1/token up to d_k.

---

## DeltaNet:  S = Σᵢ kᵢ ⊗ eᵢ   (store the ERROR eᵢ, not the raw vᵢ)

Update (β=1):  `S_t = (I − k_t k_tᵀ) S_{t-1} + k_t v_tᵀ`, which expands to `S += k_t ⊗ e_t` with
```
e₁ = v₁
e₂ = v₂ − (k₂·k₁) v₁
e₃ = v₃ − (k₃·k₁) v₁ − (k₃·k₂) e₂          (general: eᵢ = vᵢ − Σ_{l<i}(kᵢ·kₗ)eₗ)
```

**1 token** — identical to plain (nothing to correct yet):
```
      ⎡ k11·v11   k11·v12   k11·v13 ⎤
S  =  ⎢ k12·v11   k12·v12   k12·v13 ⎥
      ⎣ k13·v11   k13·v12   k13·v13 ⎦
```
**2 tokens** (`S += k₂ ⊗ e₂`, `e₂ = v₂ − (k₂·k₁)v₁`):
```
      ⎡ k11·v11 + k21·e21   k11·v12 + k21·e22   k11·v13 + k21·e23 ⎤
S  =  ⎢ k12·v11 + k22·e21   k12·v12 + k22·e22   k12·v13 + k22·e23 ⎥   e2c = v2c − (k₂·k₁)v1c
      ⎣ k13·v11 + k23·e21   k13·v12 + k23·e23   k13·v13 + k23·e23 ⎦
```
**3 tokens** (`S += k₃ ⊗ e₃`): general entry `S[r][c] = k1r·v1c + k2r·e2c + k3r·e3c`,
`e3c = v3c − (k₃·k₁)v1c − (k₃·k₂)e2c`.

---

## The update-form equivalence (why the e-form = the matrix form)

`(I − kkᵀ)S₁ = S₁ − kkᵀS₁` (distribute; `I·S₁ = S₁`). Then for token 2 (`S₁ = k₁v₁ᵀ`):
```
S₂ = (I − k₂k₂ᵀ)S₁ + k₂v₂ᵀ
   = k₁v₁ᵀ − k₂(k₂ᵀk₁)v₁ᵀ + k₂v₂ᵀ            (k₂ᵀk₁ = (k₂·k₁), scalar; associativity)
   = k₁⊗v₁ + k₂⊗[v₂ − (k₂·k₁)v₁] = k₁⊗v₁ + k₂⊗e₂
```

`I` is the **identity** matrix (diagonal 1, else 0), size `d_k×d_k`:
```
    ⎡1 0 0⎤          ⎡k1·k1 k1·k2 k1·k3⎤          ⎡1−k1²  −k1k2  −k1k3⎤
I = ⎢0 1 0⎥   kkᵀ = ⎢k2·k1 k2·k2 k2·k3⎥   I−kkᵀ= ⎢−k2k1  1−k2²  −k2k3⎥
    ⎣0 0 1⎦          ⎣k3·k1 k3·k2 k3·k3⎦          ⎣−k3k1  −k3k2  1−k3²⎦
```
(Convention: `S` is `d_k×d_v`, readout `y=qᵀS`, update `(I−kkᵀ)S+kvᵀ`. The transpose layout
gives `S(I−kkᵀ)+vkᵀ` — same computation.)

---

## DeltaNet S in k and v only (no e) — verified by `verify.py`

**2 tokens:**  `S[r][c] = k1r·v1c + k2r·v2c − (k₂·k₁)·k2r·v1c`

**3 tokens (general entry):**
```
S[r][c] =  k1r·v1c + k2r·v2c + k3r·v3c                     (plain part)
         − (k₂·k₁)·k2r·v1c                                  (tok2 removes tok1 overlap)
         − (k₃·k₂)·k3r·v2c                                  (tok3 removes tok2 overlap)
         − [(k₃·k₁) − (k₃·k₂)(k₂·k₁)]·k3r·v1c               (tok3 removes tok1 overlap, chain-corrected)
```
The `(k₃·k₁) − (k₃·k₂)(k₂·k₁)` coefficient is the `−L + L²` chain term from `(I+L)⁻¹`.
Orthogonal keys → all `(kᵢ·kⱼ)=0` → collapses to the plain sum → DeltaNet = linear.
