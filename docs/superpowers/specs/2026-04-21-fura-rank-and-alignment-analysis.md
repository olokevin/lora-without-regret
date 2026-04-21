# FuRA Rank and Alignment Analysis

**Date:** 2026-04-21
**Scope:** First-peek empirical analysis supporting the motivation section of the FuRA paper. Produces one paper-ready 3×3 figure plus a long-form data-notes markdown from three existing RL checkpoints (Full FT, FuRA-blocktt, SVD) on Qwen3-1.7B. Analysis-only — no new training runs. Deferred second-peek (raw-gradient measurement via one base-model forward-backward pass) is explicitly out of scope and opens a follow-up spec if triggered.

Related but distinct: `docs/superpowers/specs/2026-04-21-fura-system-eval.md` covers system-performance evaluation (wall-clock, memory, throughput). This doc covers scientific analysis of trained weights.

---

## 1. Purpose

Substantiate two claims used as empirical motivation in the FuRA paper:

1. **Different layers need different update ranks.** The stable rank of `ΔW = W_ckpt − W_base` varies by ~5–20× across layers and module types → any fixed-rank LoRA is suboptimal; FuRA's full-rank-flexible parameterisation is needed.
2. **The final update aligns with the pretrained column space.** The cumulative update `ΔW_t` sits inside `col(W_base)` at a fraction well above the random baseline. Under first peek (this spec), we establish the "update aligns" half directly. The "raw gradient has orthogonal energy" half is deferred.

Additional cross-parameterisation claim via the BlockTT/SVD rows: with subspace-constrained parameterisations, each layer naturally uses a different effective rank inside its budget, and singular vectors are selectively updated — not concentrated on top-k principal directions, not on orthogonal components either.

---

## 2. Data

Three existing RL checkpoints on Qwen3-1.7B. No new training runs.

| Method | Checkpoint root | Steps saved | Storage |
|--------|-----------------|-------------|---------|
| Full FT | `/data/yequan/fura/rl_runs/full/full-adamw-lr_2e-5-0325-215533/` | 1, 10, 50 | dense `nn.Linear` weights |
| FuRA (blocktt, s_to_frozen) | `/data/yequan/fura/rl_runs/blocktt/blocktt-adamw-lr_1e-4-output_one_block-s_to_frozen-train_small-0317-150342/` | 1, 10, 50 | `btt_l / btt_r / btt_s` cores |
| SVD (s_to_keep, train_input) | `/data/yequan/fura/rl_runs/svd/svd-adamw-lr_1e-5-s_to_keep-train_input-0317-141139/` | 1, 10, 50 | `svd_a / svd_b / svd_s` cores |

Base model: `Qwen/Qwen3-1.7B` (hidden 2048, intermediate 6144, 28 transformer blocks).

Target modules (per layer): `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

---

## 3. Metric definitions

Three metric families, one per figure column. All computed per `(method, step, layer, module)` from `W_base`, `W_ckpt`, and `ΔW = W_ckpt − W_base`.

### 3.1 Column 1 — per-layer stable rank of `ΔW`

```
σ = svdvals(ΔW.float())                  # descending, length r₀ = min(d_out, d_in)
stable_rank = (Σ σ_i²) / σ_1²             # = ‖ΔW‖_F² / ‖ΔW‖_2²
```

Stable rank chosen over entropy rank or threshold rank because (i) no threshold hyperparameter, (ii) single scalar per layer/module, (iii) standard in the subspace literature. Entropy rank and threshold-0.01 rank are computed alongside for appendix robustness but not plotted in the main figure.

### 3.2 Column 2 — update-energy alignment with pretrained column space

Let `W_base = U₀ S₀ V₀ᵀ` be the thin SVD of the base weight. Define two alignment ratios:

```
align_ΔW_U(t, l, m) = ‖U₀ U₀ᵀ ΔW_t‖_F² / ‖ΔW_t‖_F²
align_ΔW_V(t, l, m) = ‖ΔW_t V₀ V₀ᵀ‖_F² / ‖ΔW_t‖_F²
```

Both live in `[0, 1]`. Low → update has substantial energy outside `col(W_base)` (U-side) or `row(W_base)` (V-side). High → update respects the pretrained subspace.

**Which side per module (Qwen3-1.7B shapes, r₀ = min(d_out, d_in) = 2048):**

| Module | Shape `(d_out, d_in)` | U-side informative? | V-side informative? | Used in main fig |
|--------|-----------------------|---------------------|---------------------|------------------|
| q/k/v/o_proj | (2048, 2048) | No (U₀ full basis) | No | **excluded** |
| gate_proj | (6144, 2048) | Yes (4096-dim null in out) | No | **U-side** |
| up_proj | (6144, 2048) | Yes (4096-dim null in out) | No | **U-side** |
| down_proj | (2048, 6144) | No | Yes (4096-dim null in in) | **V-side** |

Only 3 of 7 modules contribute to column 2. The caption states why explicitly.

**Random baseline.** If `ΔW` were a random matrix (i.i.d. entries) with no preference for `col(W_base)` or `row(W_base)`, `E[align_ΔW_U] = r₀ / d_out` (U-side) and `E[align_ΔW_V] = r₀ / d_in` (V-side). For Qwen3-1.7B: gate/up U-side → `2048/6144 ≈ 0.333`; down V-side → `2048/6144 ≈ 0.333`. Same number for all 3 modules in column 2, drawn as a single dashed reference line per panel.

**Robustness (appendix, not main fig).** Repeat with truncated projector `U₀,k U₀,kᵀ` for `k ∈ {32, 64, 128, 256, 512}`. This reintroduces the square attention modules (for them, top-k is informative even though full-U is trivially 1.0).

### 3.3 Column 3 — per-pretrained-direction update energy heatmap

```
A = U₀ᵀ ΔW V₀                            # shape (r₀, r₀)
d_i = A[i, i]                             # diagonal
e(l, m, i) = d_i² / Σⱼ d_j²               # normalised per (l, m)
```

Plot as heatmap, `y = layer 0..27`, `x = σ-index 1..512` (truncated — tail beyond 512 is near-zero and hurts contrast), colour = `log e(l, m, i)` on a shared colormap.

Main figure shows `up_proj` and `down_proj` side by side per cell (representative tall and wide shapes). Remaining modules' heatmaps are written to the data-notes markdown as CSV dumps.

### 3.4 Additional diagnostics (collected, not plotted in main fig)

Computed during the same sweep, stored in the per-method JSON cache:
- Entropy rank, threshold-0.01 rank (col-1 robustness).
- Top-k alignment `align_ΔW_{U,V}^k` for `k ∈ {32, 64, 128, 256, 512}` (col-2 robustness).
- Frobenius norm `‖ΔW‖_F` and ratio `‖ΔW‖_F / ‖W_base‖_F` (magnitude sanity).
- Full singular spectrum of `ΔW` (feeds col-1 and an optional appendix figure).
- Principal-mass fractions: `(top_quarter, mid_half, tail_quarter)` of `e(l,m,i)` by σ-index.
- Top-10 σ-indices by energy per `(method, module)` (identifies which pretrained directions receive most update mass).
- Principal-angles between `U₀` and the top-r directions of `ΔW`'s own SVD.

---

## 4. Figure design

**File:** `docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf` (plus `.png` preview).

**Grid:** 3 rows × 3 columns. Rows = methods (Full FT, FuRA, SVD). Columns = stable rank, alignment trajectory, per-direction heatmap. Row-to-row y-axes shared per column so the reader can read the "FuRA preserves rank heterogeneity" and "all three align comparably" stories directly.

```
         Col 1 (stable rank)       Col 2 (U/V alignment)       Col 3 (heatmap selectivity)
         step=50, lines per        t ∈ {1,10,50},              step=50, representative
         module                    3 lines (gate/up/down)      modules
         ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
Row 1    │ (a) Full FT          │  │ (b) Full FT          │  │ (c) Full FT          │
Full FT  │ x=layer idx 0..27    │  │ x=step (log: 1,10,50)│  │ 2 heatmaps side by   │
         │ y=stable rank(ΔW)    │  │ y=align_ΔW ∈[0,1]    │  │   side: up, down     │
         │   (log)              │  │ lines: gate(U),      │  │ y=layer 0..27        │
         │ 7 lines, colours:    │  │   up(U), down(V)     │  │ x=σ-index 1..512     │
         │   q,k,v,o=warm       │  │ shaded ±1σ over      │  │ log-scale viridis    │
         │   gate,up,down=cool  │  │   layers per line    │  │                      │
         │                      │  │ dashed baseline 0.333│  │                      │
         ├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
Row 2    │ (d) FuRA (blocktt)   │  │ (e) FuRA             │  │ (f) FuRA             │
FuRA     │ same axes            │  │ same axes            │  │ same modules, axes   │
         │ ΔW from              │  │ ΔW from              │  │                      │
         │ materialize_blocktt  │  │ materialize_blocktt  │  │                      │
         ├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
Row 3    │ (g) SVD              │  │ (h) SVD              │  │ (i) SVD              │
SVD      │ same                 │  │ same                 │  │ same                 │
         └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

**Aesthetic defaults:**
- Dimensions: ~13 in × 9 in total (NeurIPS full-page-width).
- Col 1/2 panels: ~4:3 aspect, line plots.
- Col 3 panels: ~1:1 aspect each sub-heatmap (two side by side per panel).
- Single shared colorbar on the right of row 1, col 3, log scale.
- Legend on panel (a) only; shared across rows. Legend on panel (b) only for column 2.
- Font: matplotlib default (Computer Modern via `usetex=False` to avoid build deps).

**Caption template** (auto-filled from `fig_data.json`):
> Figure X. **Update structure across three parameterisations on Qwen3-1.7B RL (step=50).** Columns: (1) stable rank of `ΔW` per layer; (2) update-energy alignment with the pretrained column space — `U`-side projection for gate/up_proj, `V`-side for down_proj; square attention modules excluded because `U₀` spans `ℝ^{d_out}`; (3) per-pretrained-σ-direction update-energy heatmap for up_proj and down_proj. Rows: Full FT (a-c), FuRA-blocktt (d-f), SVD (g-i). Dashed line in column 2 = random-baseline alignment ratio `r₀ / max(d_out, d_in) = 0.333`.

---

## 5. Data-notes markdown

**File:** `docs/26_nips_fura_paper/rank_and_alignment_notes.md`. Auto-generated by the plotting script from the same JSON caches; idempotent.

**Structure:**

```
# Rank and Alignment: Full Metric Tables
_Auto-generated on {ISO date} by analysis/plot_rank_alignment_3x3.py_

## 1. Setup
Base model, ckpt paths, modules, layers, SVD precision.

## 2. Column-1 metrics — per-layer stable rank of ΔW (step=50)
- 2.1 Full FT  (layer × module table)
- 2.2 FuRA     (layer × module table)
- 2.3 SVD      (layer × module table)
- 2.4 Summary stats per (method, module): median, IQR, min, max, argmin/argmax layer

## 3. Column-2 metrics — U/V alignment trajectory
- 3.1 Main table: 3 modules × 3 methods × 3 steps — mean/median/per-layer alignment
- 3.2 Per-layer breakdown (appendix)
- 3.3 Top-k robustness: k ∈ {32,64,128,256,512} across all 7 modules
      (this re-introduces q/k/v/o which are trivial under full-U)

## 4. Column-3 metrics — per-direction update energy
- 4.1 Top-10 σ-indices by energy per (method, module)
- 4.2 Principal-mass / mid-mass / tail-mass fractions per (method, layer, module)
- 4.3 Layer-averaged heatmap CSV dumps (one CSV per method × module)

## 5. Additional diagnostics
- 5.1 ΔW Frobenius norm per (layer, module)
- 5.2 ‖ΔW‖_F / ‖W_base‖_F
- 5.3 Singular spectrum of W_base
- 5.4 Singular spectrum of ΔW
- 5.5 Principal angles between U₀ and U_ΔW (top-10 per layer)

## 6. Caveats
- No raw-gradient measurement in this first peek.
- Only step=1/10/50 available; no dense trajectory.
- SVD parameterisation (additive vs replacement) verified in loader at import.

## 7. Raw artifacts
Paths to per-method JSON caches and figure-data JSON.
```

---

## 6. Code architecture

### 6.1 File layout

```
analysis/
├── analyze_weights.py                      # EXISTING — reuse materialize_* helpers
├── rank_alignment_metrics.py               # NEW — pure metric functions (no I/O)
├── rank_alignment_loader.py                # NEW — ckpt loading + ΔW materialisation
├── analyze_rank_alignment.py               # NEW — entry point: compute + cache
└── plot_rank_alignment_3x3.py              # NEW — entry point: cache → figure + md

tests/
└── test_rank_alignment_metrics.py          # NEW — unit tests on synthetic tensors
```

**Split rationale.** Pure metric functions separate from I/O so they are easy to unit-test. Two entry points (compute vs plot) because metric computation is the expensive step (~5–15 min of SVDs) and plotting iterates many times during paper writing.

### 6.2 `analysis/rank_alignment_metrics.py`

Pure tensor → scalar/array functions. No I/O, no globals.

```python
def stable_rank(delta_W: torch.Tensor) -> float
def entropy_rank(delta_W: torch.Tensor) -> float
def threshold_rank(delta_W: torch.Tensor, frac: float = 0.01) -> int
def align_energy_u(delta_W: torch.Tensor, U0: torch.Tensor) -> float
def align_energy_v(delta_W: torch.Tensor, V0: torch.Tensor) -> float
def align_energy_topk(delta_W, U0_or_V0, k: int, side: Literal["U", "V"]) -> float
    # Internally truncates the full basis to its first `k` columns and applies
    # the same formula as align_energy_{u,v}. Caller passes the full U0/V0; the
    # function handles truncation so batched topk-sweeps reuse one SVD.
def per_direction_energy(delta_W, U0, V0) -> np.ndarray  # shape (r0,)
def principal_mass_fractions(energy: np.ndarray, r0: int) -> tuple[float, float, float]
def random_baseline_align(d_out: int, d_in: int, side: str) -> float
def svdvals(W: torch.Tensor) -> torch.Tensor  # thin wrapper for float32 + device consistency
```

Every function operates on one `(ΔW, U₀, V₀)` triple. Aggregation across layers/modules lives in entry-point scripts, not here.

### 6.3 `analysis/rank_alignment_loader.py`

Unifies the three parameterisations behind one interface.

```python
def load_base_svd(base_model: str, device: torch.device) -> dict[(int, str), (U0, V0)]:
    """One-time SVD of every target-module W_base. Cached at
    analysis_cache/rank_alignment/base_svd.safetensors. Skip if fresh."""

def load_delta_w(ckpt_dir: Path,
                 method: Literal["full", "blocktt", "svd"],
                 layer: int,
                 module: str,
                 base_W: torch.Tensor,
                 device: torch.device) -> torch.Tensor:
    """Return ΔW for this (ckpt, layer, module) as a dense tensor on `device`.
       - full:    load ckpt weight - base_W
       - blocktt: materialize_blocktt_weight(btt_l, btt_r, btt_s) - base_W
       - svd:     materialize_svd_weight(svd_a, svd_b, svd_s) [+/- base_W, see below]
    """
```

**SVD parameterisation detection.** `svd_layer.py:SVDLayer.forward` is read at loader import time to determine whether the SVD layer stores *the full reconstructed weight* (`W = svd_a · diag(s) · svd_b`) or *the delta only* (`W = W_base + svd_a · diag(s) · svd_b`). The loader picks the matching formula. If the pattern is ambiguous, the loader raises a clear error naming the constant to set manually.

**Per-ckpt materialisation cache.** After first materialisation, write `analysis_cache/rank_alignment/<method>_step<N>_deltas.safetensors`, keyed by `(layer, module)`. Re-runs skip materialisation and go straight to metrics.

### 6.4 `analysis/analyze_rank_alignment.py` — entry point 1

```
CLI:
  --base-model Qwen/Qwen3-1.7B
  --method {full, blocktt, svd, all}
  --ckpt-dir PATH          (required unless method=all, then hardcoded §2 paths)
  --steps 1,10,50
  --out-dir analysis_cache/rank_alignment
  --device cuda:0
  --topk-list 32,64,128,256,512
```

**Flow:**
1. Load base SVDs (cached).
2. Preflight: check all ckpt dirs/steps exist; run one (layer=0, up_proj, method=full, step=50) triple and print its metric dict.
3. For each `(method, step, layer, module)`: load `ΔW`, compute the full metric suite (including all robustness variants), append to `results[method][step][layer][module]`.
4. Write `{method}.json` per method.
5. Print quick-read summary: stable-rank range per method, mean±σ alignment per method/module, random-baseline reminder.

Runtime: ~5–15 min on H100 across all three methods. Idempotent — re-runs hit caches.

### 6.5 `analysis/plot_rank_alignment_3x3.py` — entry point 2

```
CLI:
  --cache-dir analysis_cache/rank_alignment
  --out-pdf  docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf
  --out-md   docs/26_nips_fura_paper/rank_and_alignment_notes.md
```

**Flow:**
1. Load three JSON caches.
2. Build 3×3 matplotlib figure per §4.
3. Write PDF + PNG + `rank_and_alignment_3x3.data.json` (the aggregated numbers used in the figure, so the caption can be regenerated without re-parsing the figure).
4. Write `rank_and_alignment_notes.md` per §5.
5. No GPU, < 30 s.

### 6.6 `tests/test_rank_alignment_metrics.py`

Synthetic-tensor unit tests. No real ckpts.

- `stable_rank`: identity matrix → rank; rank-1 matrix `uvᵀ` → 1.
- `align_energy_u`: constructed `ΔW ∈ col(U₀)` → 1.0; constructed `ΔW ⊥ col(U₀)` → 0.0; constructed mixture `αA + βB` → `‖αA‖²/(‖αA‖² + ‖βB‖²)`.
- `align_energy_v`: symmetric tests on the V side.
- `per_direction_energy`: `ΔW = U₀ diag(c) V₀ᵀ` → returns `c² / ‖c‖²`.
- `random_baseline_align`: analytic `r₀ / max(d_out, d_in)` plus Monte-Carlo empirical check to 3 decimals.

No tests against real ckpts (loader bugs surface on first entry-point run; that's acceptable for research code).

### 6.7 What is *not* built

- No new training runs.
- No changes to `btt_layer.py`, `svd_layer.py`, `run_rl.py`, or any training script.
- No CI gate.
- No interactive plotting; PDF only.

---

## 7. Error handling and edge cases

**Zero-update layers.** If `‖ΔW‖_F < 1e-8`, return `NaN` for all metrics. NaN renders as a gap in the figure — honest and visible.

**Degenerate singular values in `W_base`.** Full-subspace projections (`align_energy_u/v`) are rotation-invariant, so ties in `σ₀` don't change them. Per-direction energy `d_i` (column 3) *is* rotation-dependent inside a degenerate subspace; noted in the data-notes caveats section.

**SVD parameterisation ambiguity.** Loader reads `svd_layer.py` at import, picks formula, or raises a clear error naming the constant to set manually.

**Missing modules in a ckpt.** Log warning, record `None`, plotting renders as gap.

**Device memory.** One module at a time; peak SVD on `6144×2048` fp32 is ~200 MB; `empty_cache()` every layer (28 modules). Expected peak < 4 GB.

**Tiny step=1 `ΔW` for Full FT.** Well-defined but noisy. Scale-invariant metrics (stable rank, alignment) are unaffected; the single noisy point doesn't distort the 3-point line.

**Pre-sweep validation.** Entry point 1 verifies ckpt directories and base-model shapes, runs one test triple, prints its full metric dict before launching the sweep.

**Post-sweep assertions.** `align_ΔW_u ∈ [0, 1]` asserted for every entry. Max stable rank `< min(d_out, d_in) = 2048` asserted. `random_baseline_align` checked against analytic formula once at load.

**When the result is weak (alignment near random baseline).** Not a code bug. Recorded in §6 of the notes markdown. Decision point: open a follow-up spec for the deferred second-peek (`align_g(0)` via one base-model forward-backward pass).

---

## 8. Deliverables and execution order

### 8.1 Deliverables

| # | Artifact | Path | Produced by |
|---|----------|------|-------------|
| 1 | Metric library | `analysis/rank_alignment_metrics.py` | hand-written |
| 2 | Loader | `analysis/rank_alignment_loader.py` | hand-written |
| 3 | Unit tests | `tests/test_rank_alignment_metrics.py` | hand-written, must pass before #4 |
| 4 | Base-model SVD cache | `analysis_cache/rank_alignment/base_svd.safetensors` | first call to `analyze_rank_alignment.py` |
| 5 | Per-method JSON caches | `analysis_cache/rank_alignment/{full,blocktt,svd}.json` | `analyze_rank_alignment.py --method all` |
| 6 | Headline figure | `docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf` (+ `.png`) | `plot_rank_alignment_3x3.py` |
| 7 | Data-notes markdown | `docs/26_nips_fura_paper/rank_and_alignment_notes.md` | `plot_rank_alignment_3x3.py` (same run) |
| 8 | Figure-data JSON | `docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.data.json` | `plot_rank_alignment_3x3.py` (same run) |

### 8.2 Execution order

```bash
# 1. Tests pass first
uv run python -m pytest tests/test_rank_alignment_metrics.py -v

# 2. Compute metrics (one-time, ~5-15 min on H100)
CUDA_VISIBLE_DEVICES=0 uv run python analysis/analyze_rank_alignment.py \
    --base-model Qwen/Qwen3-1.7B \
    --method all \
    --out-dir analysis_cache/rank_alignment

# 3. Plot + markdown (< 30 s, no GPU)
uv run python analysis/plot_rank_alignment_3x3.py \
    --cache-dir analysis_cache/rank_alignment \
    --out-pdf docs/26_nips_fura_paper/figs/rank_and_alignment_3x3.pdf \
    --out-md  docs/26_nips_fura_paper/rank_and_alignment_notes.md

# 4. Read the quick-read summary printed by step 2.
#    If alignment clears 0.333 by a wide margin → figure is done.
#    If not → open follow-up spec for the deferred second-peek.
```

Both entry points idempotent. Re-runs skip fresh caches.

### 8.3 Success criteria

**Minimum (first peek done).**
- Tests green.
- 3×3 PDF renders with no NaN/zero panels.
- Console quick-read prints; post-sweep asserts pass.
- Notes markdown written with all §5 tables populated.

**Paper-ready.**
- Col 1: stable rank spans at least 5× across layers within at least one module type, for at least one method.
- Col 2: `align_ΔW_u` (gate, up) and `align_ΔW_v` (down) sit clearly above 0.333, for Full FT at step=50, averaged over layers.
- Col 3: heatmaps show non-uniform, non-monotone-in-σ pattern (neither all-principal nor all-orthogonal).

If all three hold → write caption and ship. If any hold weakly → log in §6 of the notes markdown, open follow-up spec for the deferred second-peek.

### 8.4 Commit plan

One branch `analysis/rank-alignment`. Logical commits:

1. `analysis: add pure rank-alignment metric functions + tests`
2. `analysis: add ckpt loader with blocktt/svd materialisation`
3. `analysis: add analyze_rank_alignment entry point (metric sweep + cache)`
4. `analysis: add plot_rank_alignment_3x3 (figure + notes markdown)`
5. `paper: add rank-and-alignment 3×3 figure and notes`  (artefacts only, not code)

---

## 9. Out of scope

- **New training runs** — not in this spec. First peek reuses existing step=1/10/50 ckpts.
- **Raw-gradient alignment measurement** (`align_g(0)` via base-model forward-backward) — deferred second-peek; separate follow-up spec if first-peek is inconclusive.
- **`run_rl.py` / `run_sft.py` / layer definitions** — not modified.
- **System performance** (wall-clock, memory, throughput) — covered by `docs/superpowers/specs/2026-04-21-fura-system-eval.md`.
- **Accuracy re-measurement** — read from existing paper tables if needed, not recomputed.
- **Multi-seed variance on the metrics** — single ckpt per method, no error-bar-over-seeds.
- **Dense training trajectory** — only the 3 saved ckpts per method.
