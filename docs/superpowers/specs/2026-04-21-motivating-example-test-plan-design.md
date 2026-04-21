# Motivating Example — Test Plan & Report Pipeline (Design)

**Status:** accepted, 2026-04-21.
**Scope:** Produce Figure 2 of the NeurIPS 2026 FURA paper (§3.1 + §3.4) and a machine-generated results report, for **two (model, task) pairs**:

- **RL:** Qwen3-1.7B, GRPO math — Full FT vs BTT/FURA at step=50.
- **SFT:** LLaMA-3-8B, commonsense — Full FT vs BTT/FURA (SFT full-FT run still in progress at time of spec).

**Parent plan:** `docs/26_nips_fura_paper/docs/motivating_example_design.md`.

This spec refines the parent plan into (1) a concrete set of analysis scripts, (2) test coverage for each script, and (3) the format of the final `.md` report deliverable.

---

## 1. Inputs (already on disk)

Checkpoints are `model.safetensors` (merged dense, saved via `save_merged_checkpoint` with `_build_factored_dense_state_dict`). No factor-tensor reconstruction is needed at analysis time.

| Pair | Method | Path | Base model |
|------|--------|------|------------|
| RL   | Full FT | `/data/yequan/fura/rl_runs/full/full-adamw-lr_2e-5-0420-173501/step=50/` | `Qwen/Qwen3-1.7B` |
| RL   | BTT/FURA | `/data/yequan/fura/rl_runs/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0419-185333/step=50/` | `Qwen/Qwen3-1.7B` |
| SFT  | Full FT | `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/full-lr_5e-5-seed_43/` (in-flight) | `meta-llama/Meta-Llama-3-8B` |
| SFT  | BTT/FURA | `/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_43/` | `meta-llama/Meta-Llama-3-8B` |

RL eval at step=50 (from each ckpt's `eval_results.json`):
- Full FT: MATH-500 0.636, AIME-24 0.1375, AIME-25 0.1542, AMC23 0.475.
- BTT/FURA: MATH-500 0.614, AIME-24 0.100, AIME-25 0.1292, AMC23 0.550.

(The parent plan references 0.886 accuracies that do not match these checkpoints; the report records whatever the actual numbers are. We do not re-train.)

BTT decomposition config (from run dir name): `decomp_mode=output_one_block`, `s_merged_to=keep_trainable`, `train_position=small`. `factorize_by_head` defaults to `False` in `run_rl.sh`; verify at implementation time by reading the run's wandb config, but the default is correct for the canonical runs.

Blocking scheme resolved via `btt_layer._closest_factor_pair`:

| Pair | Module | d_in | d_out | (m, a, n, b) |
|------|--------|------|-------|--------------|
| RL | q/k/v/o | 1536 | 1536 | (1, 1536, 32, 48) |
| RL | gate/up | 1536 | 8960 | (1, 8960, 32, 48) |
| RL | down | 8960 | 1536 | (1, 1536, 80, 112) |
| SFT | q/o | 4096 | 4096 | (1, 4096, 64, 64) |
| SFT | k/v (GQA) | 4096 | 1024 | (1, 1024, 64, 64) |
| SFT | gate/up | 4096 | 14336 | (1, 14336, 64, 64) |
| SFT | down | 14336 | 4096 | (1, 4096, 112, 128) |

---

## 2. Directory layout

```
docs/26_nips_fura_paper/analysis/
  _common.py              # ckpt loading, target-module iteration, blocking, block-SVD cache
  compute_panel_a.py
  compute_panel_b.py
  compute_panel_c.py
  surgical_ablation.py
  plot_motivation.py
  write_report.py
  run_all.sh              # driver for one (pair) at a time

docs/26_nips_fura_paper/results/   # committed
  qwen3_1p7b_grpo/{motivation.png, motivation.pdf, SUMMARY.md}
  llama3_8b_commonsense/{motivation.png, motivation.pdf, SUMMARY.md}

/data/yequan/fura/motivation/      # NOT committed (large)
  qwen3_1p7b_grpo/
    _svd_cache/
    full/{panel_a.csv, panel_b.npz, panel_c.csv, panel_c_spectra.npz,
          aligned_only_ckpt/, ablation_eval.json, ablation_summary.json}
    blocktt/{panel_a.csv, panel_b.npz, panel_c.csv, panel_c_spectra.npz}
  llama3_8b_commonsense/
    …

tests/
  test_motivation_common.py
  test_motivation_panels.py
  test_motivation_ablation.py
  test_motivation_plot.py
  test_motivation_report.py
```

Every compute script accepts `--base-model`, `--checkpoint`, `--artifacts-root`, `--device`, `--dtype`. `plot_motivation.py` and `write_report.py` additionally accept `--figures-dir` / `--report-dir` under `docs/26_nips_fura_paper/results/<pair>/`.

---

## 3. `_common.py` — shared helpers

### 3.1 `load_weight_pair(base_model, ckpt_dir, device) -> Iterator[WeightPair]`

Yields one linear layer at a time:

```python
@dataclass
class WeightPair:
    layer_idx: int
    module_name: str           # e.g. "model.layers.7.self_attn.q_proj"
    module_type: str           # one of: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    W_0: Tensor                # (d_out, d_in), fp32 on `device`
    W_ft: Tensor               # (d_out, d_in), fp32 on `device`
    blocking: Blocking
```

Reuses `analysis.analyze_weights.TARGET_MODULES`. Streams via `safe_open` to avoid loading the full model at once. Both ckpts are dense — no factor reconstruction.

### 3.2 `resolve_blocking(module_name, in_features, out_features, decomp_mode, factorize_by_head, model_config) -> Blocking`

```python
@dataclass
class Blocking:
    m: int
    a: int
    n: int
    b: int
    decomp_mode: str   # "output_one_block" (our config)
```

Internally calls the exact same factorization code paths as `btt_layer.convert_linear_to_btt`: `_closest_factor_pair` for the unblocked side plus head-factorization when requested. Not reimplemented — imported.

### 3.3 `block_svd(W_0, blocking, cache_dir=None) -> BlockSVD`

Per-block SVD of `W_0`, cached to `<cache_dir>/<module_name>.safetensors`. Cache key = `(module_name, blocking, sha256(W_0.view(-1)[:1024].cpu().numpy().tobytes()))` (shape + sample hash; collision risk negligible for same-base-model).

```python
@dataclass
class BlockSVD:
    U: Tensor   # (n, a, min(a,b)) — left singular vectors of each block
    S: Tensor   # (n, min(a,b))    — singular values
    V: Tensor   # (n, b, min(a,b)) — right singular vectors (note: V, not V^T)
```

For `output_one_block` with `a ≥ b` (the common case), each block has rank `b`, so `U.shape = (n, a, b)`, `S.shape = (n, b)`, `V.shape = (n, b, b)`.

### 3.4 Tests — `tests/test_motivation_common.py`

- `test_load_weight_pair_yields_target_modules_only` — synthetic 2-layer model, confirm only Q/K/V/O/gate/up/down, no embeddings/layernorms.
- `test_resolve_blocking_matches_btt_layer` — for each of the 7 module types on Qwen3-1.7B and LLaMA-3-8B configs, assert `resolve_blocking(...)` equals `BTTLayer(...)` instantiation result via `convert_linear_to_btt`.
- `test_block_svd_round_trip` — per-block reconstruction equals W_0 to fp32 precision.
- `test_block_svd_cache_hit` — second call with same key reads from disk; monkey-patch `torch.linalg.svd` to raise if invoked.

---

## 4. Per-panel compute scripts

### 4.1 `compute_panel_a.py` — effective rank

Per linear layer:
1. `ΔW = W_ft − W_0`, fp32.
2. `σ = svdvals(ΔW)`.
3. `r90 = min r : sum(σ²[:r]) ≥ 0.90 · sum(σ²)`; `r99` at 0.99.
4. `stable_rank = sum(σ²) / σ[0]²`.

**Output:** `panel_a.csv` with columns `layer_idx, layer_name, module_type, d_in, d_out, r90, r99, stable_rank, delta_fro, w0_fro`.

**Stdout:** min/median/max `r90` grouped by `module_type`; flag rows where `r90 == min(d_in, d_out)` (saturation).

### 4.2 `compute_panel_b.py` — per-direction update magnitude

Per layer, per block `k ∈ 0..n-1`:
1. `W_0_k = W_0[:, k·b:(k+1)·b]` (for `output_one_block`).
2. `(U_k, S_k, V_k) = block_svd(W_0, blocking).[U,S,V][k]` (cached).
3. `ΔW_k = ΔW[:, k·b:(k+1)·b]`.
4. `ΔΣ_k = diag(U_k.T @ ΔW_k @ V_k)`. Length `b` (or `min(a,b)`).

**Output (ragged across layer types, split by module-type group to stay rectangular):**
`panel_b.npz` with:
- `layer_names_<group>`: str array, length L_group
- `delta_sigma_<group>`: float32, shape `(L_group, n_group, b_group)`
- `sigma_w0_<group>`: float32, shape `(L_group, n_group, b_group)` — the pretrained σ (for sorting the x-axis of heatmaps and for principal/off-principal classification)

Groups: `attn_q`, `attn_kv` (handles GQA where d_out differs), `attn_o`, `mlp_gate_up`, `mlp_down`. Groups with identical `(n, b)` share one array.

**Stdout sanity:** for 3 random layers, print top-5 directions by `|ΔΣ_k|` with their rank in `σ(W_0_k)` ordering (sanity for mixed principal/off-principal claim).

### 4.3 `compute_panel_c.py` — orthogonal residual

Per layer, per block `k`:
1. `U_k` from cache.
2. `ΔW_aligned_k = U_k @ (U_k.T @ ΔW_k)`.
3. `ΔW_orth_k = ΔW_k − ΔW_aligned_k`.
4. `orth_fro_sq_k = ‖ΔW_orth_k‖²_F`, `full_fro_sq_k = ‖ΔW_k‖²_F`.
5. `orth_svdvals_k = svdvals(ΔW_orth_k)` (length `min(a-b, b)`; note if `a == b`, orth component is zero by construction — record `NaN` for `svdvals`).
6. `aligned_svdvals_k = svdvals(ΔW_aligned_k)`.

Per-layer aggregation:
- `OEF_ℓ = Σ_k orth_fro_sq_k / Σ_k full_fro_sq_k`.
- `orth_spectral_flatness = σ_1 / mean(σ)` for concatenated orth singular values; same for aligned. Flatter = more noise-like.

**Output:**
- `panel_c.csv` columns: `layer_idx, layer_name, module_type, OEF, orth_sigma1, aligned_sigma1, orth_spectral_flatness, aligned_spectral_flatness, orth_stable_rank_mean, aligned_stable_rank_mean`.
- `panel_c_spectra.npz`: per-block `orth_svdvals` and `aligned_svdvals`, split by module-type group same as panel_b.

For the BTT ckpt: OEF should be zero by construction. The script still runs and asserts `max(OEF) < 1e-5`, printing "constraint verified numerically".

**Stdout:** Full-FT OEF distribution (min/median/max) per module type.

### 4.4 Tests — `tests/test_motivation_panels.py`

Fixtures: small (d_in=16, d_out=24) fp32 random matrices on CPU.

- `test_panel_a_r90_on_rank_5_update` — ΔW of known rank 5 with decreasing singular values → r90 matches closed-form.
- `test_panel_a_stable_rank_equal_sigmas` — equal σ give `stable_rank == min(d_in, d_out)`.
- `test_panel_b_projection_identity` — ΔW = W_0 → `ΔΣ_k == σ(W_0_k)` per block.
- `test_panel_b_zero_update` — ΔW = 0 → all-zero `ΔΣ`.
- `test_panel_b_orthonormal_basis` — `U_k^T @ U_k == I`, `V_k^T @ V_k == I`.
- `test_panel_c_btt_orth_is_zero` — synthetic block-constrained update: assert `OEF < 1e-10`.
- `test_panel_c_random_oef_expectation` — random Gaussian ΔW on `a=24, b=8` block: `E[OEF] ≈ (a-b)/a = 0.667`; tolerance ±0.05 on fixed seed.
- `test_block_svd_cache_shared_between_panels_b_and_c` — sequential runs use the same cached SVD; mock-verify the second call doesn't recompute.

End-to-end smoke (under `@unittest.skipUnless(os.environ.get('MOTIVATION_SMOKE'))`, heavy):
- `test_panels_a_b_c_on_tiny_qwen3` — uses `tests/smoke_runs/blocktt_eval_smoke/tiny_qwen3_model/` + hand-constructed tiny "trained" ckpt with known ΔW; assert all three panels produce well-formed outputs in < 30 s.

---

## 5. `surgical_ablation.py`

Builds the aligned-only model and evaluates it (Full FT only).

1. Stream each target linear layer from the Full FT ckpt + base model.
2. For each layer, per block: `W_aligned_k = W_0_k + U_k @ (U_k.T @ (W_ft_k − W_0_k))`.
3. Stitch blocks back into `(d_out, d_in)` weight; preserve non-target layers (embeddings, layernorms, etc.) exactly from the Full FT ckpt.
4. Write `<artifacts-root>/full/aligned_only_ckpt/` as a standard HF merged directory (safetensors, config, tokenizer copied).
5. Evaluate:
   - **RL:** `eval_rl.py --checkpoint <aligned_only_ckpt_dir> --output-json <artifacts-root>/full/ablation_eval.json`.
   - **SFT:** delegate to the existing commonsense eval harness — exact command resolved at implementation time (the ckpt dir has an `err.log` / `commonsense/` subdir suggesting an existing script). If not readily wired, emit the ckpt + a `HOW_TO_EVAL.md` stub and mark ablation verdict `"PARTIAL (eval pending)"`.
6. Write `<artifacts-root>/full/ablation_summary.json` — a post-processing pass over `ablation_eval.json` + the two training-time `eval_results.json` files:

```jsonc
// Example shape. Values filled in from real eval outputs.
{
  "full_ft_acc":      {"MATH-500": 0.636, "AIME-24": 0.138, "AIME-25": 0.154, "AMC23": 0.475},
  "aligned_only_acc": {"MATH-500": 0.0,   "AIME-24": 0.0,   "AIME-25": 0.0,   "AMC23": 0.0  },
  "fura_acc":         {"MATH-500": 0.614, "AIME-24": 0.100, "AIME-25": 0.129, "AMC23": 0.550},
  "delta_aligned_vs_full": {"MATH-500": 0.0, "AIME-24": 0.0, "AIME-25": 0.0, "AMC23": 0.0},
  "primary_metric": "MATH-500",
  "verdict_threshold_pass":    0.005,
  "verdict_threshold_partial": 0.015
}
```

Optional `--amplify-alpha 0.5,1.0,1.5,2.0` sweep (off by default; each α is a full eval run).

### Tests — `tests/test_motivation_ablation.py`

- `test_aligned_only_trivial_when_update_in_subspace` — ΔW crafted inside col(U_k) → `W_aligned == W_ft` to fp32 precision.
- `test_aligned_only_roundtrips` — produced ckpt loads via `AutoModelForCausalLM.from_pretrained` on `tests/smoke_runs/blocktt_eval_smoke/tiny_qwen3_model/` without errors.
- `test_stitching_block_layout` — per-block projection + stitching equals a direct dense computation on (in=24, out=16).
- `test_ablation_summary_deltas` — given synthetic Full-FT / aligned / FURA eval JSONs, verdict thresholds classified correctly at each boundary.

---

## 6. `plot_motivation.py`

Reads `/data/yequan/fura/motivation/<pair>/{full,blocktt}/*.{csv,npz}` and ablation JSON; writes `docs/26_nips_fura_paper/results/<pair>/motivation.{png,pdf}`.

2×3 grid, matplotlib, serif 10pt, width 6.75in, `dpi=300`.

| Row | Left (Full FT)                                | Right (FURA)                                           |
|-----|-----------------------------------------------|--------------------------------------------------------|
| a   | r90 per layer, bar, colored by module type    | r90 per layer, same                                    |
| b   | `|ΔΣ_k|` heatmap, log color                   | `|ΔΣ_k|` heatmap, same colormap                         |
| c   | per-layer OEF bar                             | 3-bar test acc: Full FT / Aligned-Only / FURA          |

Horizontal dashed lines at r=64 and r=128 on row (a). Single top legend. Consistent colors: Full FT `#1f77b4`, FURA `#d62728`, Aligned-Only `#2ca02c`.

### Tests — `tests/test_motivation_plot.py`

- `test_plot_generates_png` — synthetic inputs in tmpdir → output file exists, nonzero bytes.
- `test_plot_handles_missing_ablation` — absent `ablation_summary.json` → row (c) right annotated "eval pending", no crash.
- `test_plot_shared_axes` — row (a) left/right y-limits equal.

No pixel-level comparisons. Rendering correctness is visual review.

---

## 7. `write_report.py`

Renders `SUMMARY.md` from all artifacts.

### 7.1 Sections

1. Header with ckpt paths + eval numbers.
2. **Verdict table** (3 rows: panels a/b/c, each PASS/PARTIAL/FAIL).
3. **§3.1 camera-ready paragraph drafts** — three paragraphs mirroring the paper's structure, real numbers templated in via Jinja2.
4. **§3.4 camera-ready paragraph drafts** — same.
5. **Figure inline:** `![Figure 2](motivation.png)`.
6. **Engineering appendix:**
   - Resolved blocking table.
   - Top-20 layers by r90 (head + tail of panel_a.csv).
   - Full commands to reproduce.
   - Caveats (e.g. eval mismatch with paper's 0.886, SFT ablation wiring status, `factorize_by_head` flag confirmed).

### 7.2 Verdict thresholds (derived from parent plan's "Success Flags")

- **Panel (a):** PASS if `r90_max / max(r90_min, 1) ≥ 10` and `spearman(r90_full, r90_fura) ≥ 0.7`; PARTIAL if ratio ≥ 3; else FAIL.
- **Panel (b):** PASS if mean-over-layers (energy in top-5% directions / total energy) ≥ 0.5 AND at least 50% of layers have `|spearman(|ΔΣ|, rank(σ))| ≤ 0.3` (mixed principal/off-principal) AND `spearman(|ΔΣ|_fura, |ΔΣ|_full) ≥ 0.5`; PARTIAL if non-uniformity holds but mixed/agreement fails; else FAIL.
- **Panel (c):** PASS if mean OEF ≥ 0.05 AND `aligned_acc - full_ft_acc ≥ -0.005`; PARTIAL if OEF ≥ 0.05 AND delta ≥ -0.015; else FAIL. If ablation eval absent, verdict is `"PARTIAL (eval pending)"`.

### 7.3 Tests — `tests/test_motivation_report.py`

- `test_report_renders_with_all_inputs` — fixture CSVs/NPZ/JSON → output has all section headers, no unrendered `{{ }}`.
- `test_verdict_thresholds_parametrized` — each PASS/PARTIAL/FAIL boundary for all three panels.
- `test_report_handles_missing_ablation` — panel-c verdict = `"PARTIAL (eval pending)"`, dedicated note in the appendix.
- `test_report_numbers_match_inputs` — sanity: at least one numeric string from each CSV appears verbatim in the rendered `.md`.

---

## 8. Driver — `run_all.sh`

```bash
#!/usr/bin/env bash
# Usage: run_all.sh <pair> [--skip-ablation]
# pair ∈ {qwen3_1p7b_grpo, llama3_8b_commonsense}
set -euo pipefail

PAIR="$1"; shift || true
SKIP_ABLATION=0
for arg in "$@"; do [ "$arg" = "--skip-ablation" ] && SKIP_ABLATION=1; done

case "$PAIR" in
  qwen3_1p7b_grpo)
    BASE=Qwen/Qwen3-1.7B
    FULL=/data/yequan/fura/rl_runs/full/full-adamw-lr_2e-5-0420-173501/step=50
    BTT=/data/yequan/fura/rl_runs/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_keep_trainable-train_small-0419-185333/step=50
    ;;
  llama3_8b_commonsense)
    BASE=meta-llama/Meta-Llama-3-8B
    FULL=/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/full-lr_5e-5-seed_43
    BTT=/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/blocktt-lr_2e-4-decomp_output_one_block_pos_small_smerge_keep_trainable-seed_43
    ;;
  *) echo "unknown pair: $PAIR" >&2; exit 2 ;;
esac

ART=/data/yequan/fura/motivation/$PAIR
RES=docs/26_nips_fura_paper/results/$PAIR

for METHOD in full blocktt; do
  CKPT=$([ "$METHOD" = "full" ] && echo $FULL || echo $BTT)
  python docs/26_nips_fura_paper/analysis/compute_panel_a.py --base-model $BASE --checkpoint $CKPT --artifacts-root $ART/$METHOD
  python docs/26_nips_fura_paper/analysis/compute_panel_b.py --base-model $BASE --checkpoint $CKPT --artifacts-root $ART/$METHOD
  python docs/26_nips_fura_paper/analysis/compute_panel_c.py --base-model $BASE --checkpoint $CKPT --artifacts-root $ART/$METHOD
done

if [ "$SKIP_ABLATION" = "0" ]; then
  python docs/26_nips_fura_paper/analysis/surgical_ablation.py --base-model $BASE --checkpoint $FULL --artifacts-root $ART
fi

python docs/26_nips_fura_paper/analysis/plot_motivation.py --artifacts-root $ART --figures-dir $RES
python docs/26_nips_fura_paper/analysis/write_report.py     --artifacts-root $ART --report-dir   $RES --pair $PAIR
```

Keeps the two pairs parallelizable via separate invocations.

---

## 9. Execution cost estimates (single H100)

| Step | Qwen3-1.7B | LLaMA-3-8B |
|------|-----------|------------|
| panel_a | ~3 min | ~15 min |
| panel_b (first run, block SVD cold) | ~2 min | ~20 min |
| panel_c (cache warm) | ~2 min | ~10 min |
| surgical ablation ckpt build | ~5 min | ~15 min |
| ablation eval | ~15 min (4 math sets) | harness-dependent |
| plot + report | seconds | seconds |

Total per pair: ~30 min Qwen3, ~1 hr LLaMA-3 (plus SFT eval time).

---

## 10. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| `factorize_by_head` differs from default → wrong `(m,n,a,b)` | Read wandb config of the canonical runs in `_common.resolve_blocking`; assert blocking matches the resolved values from §1. |
| Paper's "0.886" baseline doesn't match our 0.636/0.614 | Report uses real numbers; caveat in appendix. Narrative claims unaffected since verdicts use per-metric deltas, not absolute accuracy. |
| SFT eval harness not yet plumbed | `surgical_ablation.py` emits ckpt + stub; verdict degrades to `"PARTIAL (eval pending)"`; re-run `write_report.py` once eval lands. |
| OEF negligible (< 2%) because `a ≈ b` shrinks the orthogonal subspace. Per §1 blocking table: Qwen3 attn a/b = 32, LLaMA-3 attn a/b = 64, FFN ratios all ≥ 32 — so under random-ΔW baseline OEF ≈ (a−b)/a ≥ 0.97. Risk applies only if Full-FT ΔW happens to concentrate within `col(U_k)` for non-random reasons. | Pre-check printed by panel_c. If triggered, reframe panel (c) right column per parent plan §Risk table. |
| FURA rank profile doesn't correlate with Full FT (panel a) | Verdict threshold allows PARTIAL at ≥ 3× variation; report records Spearman and Pearson; paper wording can soften. |

---

## 11. Deliverables

**Committed to git:**
- `docs/26_nips_fura_paper/analysis/*.py` (7 files) + `run_all.sh`.
- `docs/26_nips_fura_paper/results/qwen3_1p7b_grpo/{motivation.png, motivation.pdf, SUMMARY.md}`.
- `docs/26_nips_fura_paper/results/llama3_8b_commonsense/{motivation.png, motivation.pdf, SUMMARY.md}` (gated on SFT full-FT completion).
- `tests/test_motivation_*.py` (5 files).

**Not committed:**
- `/data/yequan/fura/motivation/**` (CSVs, NPZs, aligned-only ckpts, eval JSONs).

**Paper integration:** `neurips_2026.tex` §3.1 and §3.4 paragraphs pull final numbers from the generated `SUMMARY.md` camera-ready drafts.
