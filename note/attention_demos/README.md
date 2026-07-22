# Attention-variants demos (GDN / MLA / MHA)

Scripts behind `../note_attention_variants_gdn_mla.md` and the `../../h100_setup/*.png` figures.
All are standalone. Run over the devserver proxy (matplotlib/numpy not installed locally):

```
export https_proxy=fwdproxy:8080 http_proxy=fwdproxy:8080
cd /tmp && uv run --no-project --with numpy --with matplotlib python3 <script>
```

## Saved run outputs
`outputs/<script>.txt` holds the captured stdout of each numeric demo (regenerate with
the run command above). E.g. `outputs/rank_grow.txt`, `outputs/recall_bench.txt`.

## Formulas & written-out worked example
- `weight_formulas.md` — the per-token weight `wⱼ` formula for linear / softmax / DeltaNet
  (math only), including `L` = strictly-lower-tri of `KKᵀ` written out with only q,k.
- `weights.py` — computes/verifies those weights on the 3-key example → `outputs/weights.txt`.
- `formula_fig.py` → `../../h100_setup/attention_weight_formulas.png` — the three weight
  formulas TYPESET (math only), incl. the I+L matrix written out. (matches weight_formulas.md)
- `inv_fig.py` → `../../h100_setup/deltanet_inverse_matrix.png` — `(I+L)⁻¹` written out in
  key dot-products (M=4); `inv_check.py` verifies the symbolic entries vs numpy.
- `weights_fig.py` → `../../h100_setup/attention_weights.png` — bar-chart of w_j for the 3
  cases (linear leaks, softmax diffuse, delta clean) on the same 3-key example.
- `worked_example_recall.md` — full step-by-step recall calc (plain vs DeltaNet):
  interference with overlapping keys, overwrite, and the d_k=32 benchmark results.
- `worked.py`   — generates the small-dim step-by-step numbers in that doc.
- `rank_q.py`   — causal-mask rank lift + rank(S) plain vs delta (both full d_k).

## Numeric worked examples (numpy only)
- `gdn_demo.py`         — GDN associative memory: linear-attn interference vs DeltaNet
                          overwrite vs Gated-DeltaNet forgetting (note §3-GDN / §GDN).
- `gdn_vs_mha.py`       — MHA softmax readout vs GDN state readout `y=qᵀS` (why no cache).
- `recall_bench.py`     — multi-dim recall: plain vs DeltaNet vs ideal K⁺V; capacity(≤d_k),
                          recency, and overwrite. DeltaNet's win = overwrite + recent recall.
- `recall_softmax.py`   — same data + SOFTMAX attention: no capacity ceiling (retains all
                          tokens, O(M) cache), but still blends identical keys (Part 4 of doc).
- `s_feature.py`        — S should be regression map K⁺V, not correlation Σkvᵀ (interference).
- `order.py`            — linear attn: (qKᵀ)V == q(KᵀV); order irrelevant, softmax is the diff.
- `rank_lift.py`        — softmax lifts rank-≤d_k scores to full rank N (linear attn cannot).
- `mha_mla_numeric.py`  — MHA vs MLA: cache size, lossless K rebuild, absorbed==materialized.
- `mqa_to_mla.py`       — MQA-512 → MLA conversion: lossless (rank≤head_dim) vs lossy SVD (§3g).
- `mla_fold_params.py`  — folding W_UKᵀW_Q inflates params 3.6× / rank stays ≤128 (§3e).
- `mla_perhead_check.py`— per-head vs all-head param counts for the fold.
- `single_vs_multi.py`  — single-head vs MHA representational limit ([1,1,0,0] example, §7).

## Figure generators (matplotlib) → produce ../../h100_setup/*.png
- `mla_gdn_figs.py`     — attn_kvcache_growth, mla_compression, gdn_state_update, hybrid_stack
- `kv_variants.py`      — kv_variants_slopes
- `mla_absorb.py`       — mla_absorption
- `mla_capacity.py`     — mla_vs_shrink_capacity
- `mla_factor.py`       — mla_lowrank_factorization
- `block_flow.py`       — transformer_block_heads_mlp
- `mla_dims_heads2.py`  — mla_weight_dims_heads (final head-separated version)
- `mla_dims_fig.py` / `mla_dims_heads.py` — earlier drafts of the dims figure (superseded)
