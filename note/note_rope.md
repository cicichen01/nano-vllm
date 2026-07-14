# RoPE (Rotary Position Embedding) — Notes

A running summary of how RoPE works, why it's designed the way it is, and how it
shows up in serving (config knobs, KV cache). Built from first principles.

---

## 0. Context: model checkpoint formats (tangent that started this)

Besides **safetensors**, common LLM checkpoint formats:

- **PyTorch pickle** — `.pt` / `.pth` / `.bin` (`torch.save`), `.ckpt`. Can execute
  arbitrary code on load → the reason safetensors exists.
- **GGUF** (successor to GGML) — llama.cpp / Ollama, single-file, quantization-friendly.
- **ONNX** — framework-agnostic graph + weights.
- **TF**: SavedModel (`.pb`), HDF5 (`.h5`), TF checkpoints.
- **Flax/JAX**: `.msgpack`.
- Sharded index: `*.index.json` maps tensor name → shard file.

**Two kinds of checkpoint content:**
- *Inference/export* checkpoint = **weights + metadata only** (`name → tensor`,
  shapes, dtypes). No optimizer, no graph, no code. Need the modeling code to load.
- *Training* checkpoint = also optimizer state (Adam moments ≈ 2× weight size), LR
  scheduler, step counters, RNG state, GradScaler, etc.

PyTorch checkpoints do **not** store architecture/graph — you must instantiate the
`nn.Module` first, then `load_state_dict()`. ONNX / TF SavedModel are the exception
(they serialize the graph).

---

## 1. What RoPE is / the problem it solves

Attention `softmax(QKᵀ)` is permutation-invariant — no notion of order. Must inject
position. RoPE = **rotate the query and key vectors by an angle proportional to
position**, instead of adding a position vector (absolute encodings).

- Applied to **Q and K only**, right before the attention score. **Not** to V, **not**
  to the residual stream.
- No learned parameters — pure geometry.
- Absolute encoding per token, but produces **relative** behavior in the score.

---

## 2. The mechanism

Split each Q/K head vector into **2D pairs** of adjacent coords. Treat each pair as a
2D vector and rotate it by `m·θᵢ` where `m` = token position, `θᵢ` = per-pair frequency.

For a pair `(x₁, x₂)` at position `m`:
```
x₁' = x₁·cos(mθ) − x₂·sin(mθ)
x₂' = x₁·sin(mθ) + x₂·cos(mθ)
```

**"Pair" = two scalar components of ONE vector**, not two vectors. Rotating a pair
genuinely changes that 2D vector's direction.

**Where "relative" comes from:** query at pos `m` rotated by `mθ`, key at pos `n` by
`nθ`. Their dot product satisfies:
```
(R(mθ)·q) · (R(nθ)·k) = R((m−n)θ)·q · k
```
The rotations partially cancel → score depends only on `(m−n)`, the relative distance.
(If `m = n`, rotation cancels entirely → same as no RoPE. Only `m ≠ n` shifts the score.)

---

## 3. Why 2D pairs (not rotate the whole vector as one)

"Rotate a d-dim vector by a single angle" is undefined — rotation in d-D lives in
SO(d) and is defined *per plane* (needs `d(d−1)/2` angles).

RoPE requires: (1) norm-preserving, (2) `R(m)ᵀR(n) = R(n−m)`, (3) smooth in `m`
(one-parameter group `R(m)=exp(mG)`). The generator `G` (skew-symmetric) **always
block-diagonalizes into 2×2 rotation blocks**. So independent 2D rotations aren't a
hack — they're the *canonical form* of any valid rotation family. 2D is the atomic
unit where "rotate by an angle" is meaningful. Bonus: cheap (elementwise cos/sin, not
a full matrix multiply).

---

## 4. Why θ varies per pair: `θᵢ = base^(−2i/d)`

**Single shared θ fails two ways:**
- Redundancy: all pairs carry the same phase `mθ` → 128 dims encode one scalar.
- Aliasing: one frequency has period `2π/θ` → positions `m` and `m + 2π/θ` are
  indistinguishable. e.g. θ=30°/token → pos 0 and pos 12 identical.

Tension with a single frequency: fast → good local resolution but wraps quickly;
slow → long range but neighbors blur. Can't get both.

**Fix: geometric spectrum of frequencies.** Wavelengths span ~2π up to ~base·2π.
- High-freq (fast) pairs → resolve *local* offsets.
- Low-freq (slow) pairs → resolve *long-range* distance without wrapping.
- Like binary place-value / clock hands (sec/min/hour) → unique multi-scale fingerprint.

**Which pair gets which θ is arbitrary** (permutation-invariant — model learns it). What
matters is the *set/distribution* of frequencies, not the labeling.

**Why geometric, not just "distinct":** distinctness kills redundancy but not gaps.
Distance is *multiplicative* — geometric spacing = equal coverage per order of
magnitude (uniform in log-space), no gaps, no duplicates. Contrast:
- Clustered-fast (all wavelengths 6–60): nothing covers long range.
- Linear spacing: gap at short range + redundant duplicates at long range.

---

## 5. "Scales of distance" and the Goldilocks zone

A "scale" = how far apart tokens are, by order of magnitude (1–10 = local, 10–100 =
sentence, 100–1000 = paragraph, 1000–10000 = document).

Phase difference between tokens = `(m−n)·θ = Δ·θ`. Each frequency has a **Goldilocks
zone** = distances near its wavelength:
- `Δ ≪ wavelength` → angle change ≈ 0 → **invisible** (can't detect)
- `Δ ≈ wavelength` → clean fraction of a circle → **perfect**
- `Δ ≫ wavelength` → wrapped many times → **aliased/scrambled**

"Cover all scales" = have a Goldilocks zone at every order of magnitude.

**Key rule:** the **largest wavelength caps the max unambiguous distance.** A hand can
give an unwrapped reading for `Δ` only if its wavelength > `Δ`. Clustered-small
wavelengths → small common multiples → collisions appear at short distances (e.g.
wavelengths 10 and 60 both give identical readings for Δ=20 and Δ=80). Not total
collision — *aliasing on every hand at once*.

---

## 6. Content vs. position: are they entangled?

`φᵢ` = **content phase** = the angle between the query's and key's 2D sub-vectors in
that plane (pure content, from q_proj/k_proj). Complex form: `qᵢ·conj(kᵢ) =
|qᵢ||kᵢ|e^{i(α−β)}`, so `φᵢ = α − β`.

Without position, a pair contributes `|qᵢ||kᵢ|·cos(φᵢ)` — ordinary cosine similarity
(**not** just `|qᵢ||kᵢ|`; that's only the `φᵢ=0` case). Summed over pairs = plain `q·k`.

RoPE stacks the distance angle onto that phase:
```
no position:  |qᵢ||kᵢ| cos( φᵢ )
with RoPE:    |qᵢ||kᵢ| cos( φᵢ + (m−n)θᵢ )
              └content┘   └distance┘
```

**Is meaning entangled with distance?**
- In **Q/K space (the attention score): yes, deliberately.** Content compatibility gets
  phase-shifted by relative distance. We *want* "match this content, prefer it at this
  distance."
- In **content/residual/value space: no.** RoPE never touches V or the residual stream,
  so a word's *meaning* stays position-independent. Only the **routing/addressing**
  (who attends to whom) becomes distance-aware. Coupling is *relative* (`m−n`) not
  absolute.

**For a fixed distance Δ, bands play ROLES (not "dominant vs impactless"):**
- slow (wavelength ≫ Δ, ≈ unrotated) → **pure content matching**, position-blind
- resonant (wavelength ≈ Δ) → carries the **distance signal**
- fast (wavelength ≪ Δ) → washed out for this Δ (serving shorter distances)

Same physical dim changes role with Δ. Slow bands are NOT dead — they preserve content
matching. So you don't trade meaning for distance.

**Wavelengths are HARDWIRED per pair** (θᵢ fixed, not learned). What q_proj/k_proj learn
is how much magnitude/phase to load into each fixed band → the model chooses how
position-sensitive to be. Heads specialize:
- "attend to token right before me" → energy in fast pairs (fire only on neighbors)
- "find any earlier mention of X, any distance" → energy in slow pairs (position-agnostic)

---

## 7. KV cache & RoPE (nano-vllm)

Order in `qwen3.py`:
```python
q, k, v = qkv.split(...)                    # raw
q, k = self.rotary_emb(positions, q, k)     # RoPE on q, k
o = self.attn(q, k, v)                       # attn → store_kvcache(k, v, ...)
```

- **K cache stores ROTATED (post-RoPE) keys.** V stored **raw** (never rotated).
- Valid because a key's rotation `R(n)` uses its own fixed absolute position `n` →
  rotate **once** at generation, cache forever.
- Each decode step: rotate only the **current query** (cheap), dot against
  already-rotated cache; relative `(m−n)` behavior falls out at dot-product time.
- If you stored raw K, you'd re-rotate ALL cached keys every step = O(context) waste.
- Caveat: cached key is frozen at `R(n)`; schemes that *shift positions after caching*
  can't reuse it directly (rare, must store raw + re-rotate).

---

## 8. Config knobs (`config.json` / `params.json`)

- **`rope_theta`** (`base`, the 10000): base of the frequency ladder; sets the slowest
  wavelength / max reachable context. Bigger → longer context (long-ctx models use
  500k–1M). In nano-vllm it's the `base` arg to `get_rope`.
- **`head_dim` / `rotary_dim`**: `rotary_dim/2` = number of frequency pairs. nano-vllm
  asserts `rotary_dim == head_size` (full rotary only).
- **`partial_rotary_factor` / `rotary_pct`**: fraction of the head that rotates. Rotates
  the **first `rotary_dim` dims**; the tail passes through **unrotated** = dedicated
  position-free content channels. Examples: GPT-NeoX 0.25, GPT-J (64/256), Phi 0.4.
  Llama/Mistral/Qwen = 1.0 (full). So a 128-dim head with rotary_dim=100 → 50 pairs
  rotated, 14 pairs untouched.
- **`max_position_embeddings`**: trained/intended max length; the `max_position` used to
  precompute cos/sin tables; the reference point scaling stretches *from*.
- **`rope_scaling`**: long-context extension without full retraining:
  - `type`: `linear` (position interpolation), `dynamic` (dynamic NTK),
    `yarn` (stretch only low-freq bands — best), `llama3` (piecewise per-freq).
  - `factor`: extension multiple (e.g. 4.0).
  - `original_max_position_embeddings`: pre-extension length (YaRN/llama3).
  - YaRN/llama3 add band cutoffs (`beta_fast`/`beta_slow`, `low/high_freq_factor`).

Mental model: `rope_theta` + `rotary_dim` define the ladder; `partial_rotary_factor`
sets how many dims ride it; `max_position_embeddings` how far trained; `rope_scaling`
stretches it at inference.

---

## 9. Can you change `rope_theta` at inference?

- **Mechanically yes** — it's config, not a trained weight. Recomputing cos/sin tables
  is cheap. No gradient ever flowed into it.
- **But not free** — q_proj/k_proj and downstream weights learned to read the *training*
  ladder's phases. Arbitrary change = off-distribution → degradation.
- **Shorter-than-trained context** (match hardware/use case): **leave it alone.** Fewer
  positions is always in-distribution; just cap length.
- **Longer-than-trained context**: you *deliberately* scale it — the whole field of
  context extension:
  - **NTK-aware**: raise `rope_theta` (stretch slow hands, spare fast ones) → often
    usable zero-shot for 2–4×.
  - **Linear PI**: divide positions by factor.
  - **YaRN**: stretch only low-freq bands; best quality, usually light fine-tuning.
  - **Dynamic NTK**: scales `rope_theta` *as a function of the current runtime sequence
    length* — the literal "adjust theta based on inference length," zero retraining.
- Don't expect coherence beyond what the method/fine-tuning supports (zero-shot ~2–4×
  ok; 16× without fine-tuning falls apart). Use the `rope_scaling` config, don't
  hand-edit theta.
