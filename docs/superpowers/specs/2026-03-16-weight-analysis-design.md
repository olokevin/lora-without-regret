# Weight Update Analysis for SVD/BlockTT RLVR Training

## Purpose

Analyze how RLVR training modifies parameters in SVD/BlockTT decomposed models, inspired by "The Path Not Taken: RLVR Provably Learns Off the Principals" (Zhu et al.). The analysis explores the relationship between effective parameter updates and SVD structure, producing an interactive webpage to present findings.

## Architecture

Three-piece modular pipeline:

1. **`analysis/analyze_weights.py`** — loads base model + checkpoint, computes all metrics, outputs JSON
2. **`analysis/build_report.py`** — reads JSON, generates self-contained HTML report with Plotly.js
3. **`analysis/templates/report_template.html`** — HTML/JS template for the interactive report

Output goes to `analysis_results/` (gitignored).

## Metrics

### Reconstructed Dense Weight Level

For each target module (q/k/v/o/gate/up/down) in each layer, reconstruct `W_before` (base model linear weight) and `W_after` (from factored checkpoint params), compute `ΔW = W_after - W_before`:

1. **Update heatmap** — `|ΔW|` aggregated to row-wise and column-wise L2 norms. Shows which rows/cols receive most update (cf. paper's consensus ratio stripe patterns).

2. **Singular vector rotation** — For top-k singular vectors of `W_before`, compute angle between each `u_i`/`v_i` before and after training: `angle_i = arccos(clamp(|u_i^T @ u_i'|, 0, 1))` (absolute value of inner product to handle SVD sign ambiguity). X-axis = singular value index, Y-axis = angle in degrees. Reveals which principal directions rotated most.

3. **Spectrum comparison** — Overlay `σ(W_before)` and `σ(W_after)` curves. Compute normalized spectral shift: `NSS(W) = ||σ(W_after) - σ(W_before)||_2 / ||σ(W_before)||_2`.

4. **Principal angles** — Top-k subspace principal angles via `cos θ_i = σ_i(U_{before,k}^T @ U_{after,k})` (paper's Sec. 4.1, Wedin's sin-Θ theorem).

5. **Overlap with principal weights** — Construct principal weight mask `M_princ` = top-α weights by magnitude in rank-k SVD reconstruction of `W_before`. Construct update mask `M_update` = coordinates where `|ΔW_ij| > threshold`. Report overlap ratio `|M_princ ∩ M_update| / |M_update|` vs random baseline α.

6. **Spectral energy of ΔW** — SVD of the update itself. Shows whether the update is low-rank or distributed across many singular directions.

### Factored Parameter Level

7. **Per-core change heatmap** — For BlockTT: `|btt_l_after - btt_l_before|` and `|btt_r_after - btt_r_before|` visualized on the 3D tensor (sliced by block index). For SVD: `|svd_a_after - svd_a_before|` and `|svd_b_after - svd_b_before|`.

8. **Per-block Frobenius norm** — For BlockTT, compute Frobenius norm of change per (m, n) block pair. Shows which blocks the optimizer concentrated updates in.

### Summary Metrics (one scalar per layer per module)

- NSS (normalized spectral shift)
- Max principal angle (degrees)
- Overlap ratio with principal weights
- Relative update norm: `||ΔW||_F / ||W_before||_F`

## `analyze_weights.py` Specification

### CLI Interface

```
python analysis/analyze_weights.py \
  --base-model <path_or_hf_id>    # e.g. Qwen/Qwen3-1.7B
  --checkpoint <path>              # directory with safetensors
  --train-mode blocktt|svd|lora   # how to reconstruct W_after
  --output <path.json>            # default: analysis_results/metrics.json
  --top-k <int>                   # singular values for subspace analysis (default: 64)
  --principal-alpha <float>       # fraction for principal weight mask (default: 0.1)
  --layers <comma-sep>            # detailed analysis layers (default: 0,13,27)
  --update-threshold <float>      # threshold for update mask as fraction of per-module max |ΔW| (default: 0.01)
```

Note: `--layers` values are bounds-checked against the actual model layer count. All angular computations clamp `cos` values to `[-1, 1]` before `arccos` to avoid NaN from floating point.

### Processing Flow

1. Load base model weights directly from safetensors (no model instantiation, CPU only).
2. Load checkpoint weights from safetensors.
3. For each target module in each transformer layer:
   a. Extract `W_before` from base model (the original `nn.Linear` weight).
   b. Reconstruct `W_after`:
      - **BlockTT**: materialize from `btt_l` and `btt_r` cores (and `btt_s` if present, otherwise S was absorbed into cores). Infer block dimensions from tensor shapes: given `btt_l: (m, rank*n, a)` and `btt_r: (n, b, m*rank)`, extract `m = btt_l.shape[0]`, `a = btt_l.shape[2]`, `n = btt_r.shape[0]`, `b = btt_r.shape[1]`, `rank = btt_r.shape[2] // m` (equivalently `btt_l.shape[1] // n`). Materialization: reshape `btt_r` to `(n, b, m, rank)`, reshape `btt_l` to `(m, rank, n, a)`, contract over rank to get per-block weight `W_block[m,n] = btt_l[m,:,n,:] @ btt_r[n,:,m,:]` i.e. `(a, rank) @ (rank, b) -> (a, b)`, then assemble blocks into full `(m*a, n*b)` weight. If `btt_s` exists, multiply: `btt_l[m,:,n,:] * btt_s[m,n,:].unsqueeze(-1)` before contraction.
      - **SVD**: materialize from `svd_a` and `svd_b` (and `svd_s` if present). `W_after = svd_a @ svd_b` or `W_after = (svd_a * svd_s.unsqueeze(0)) @ svd_b` if `svd_s` exists.
      - **LoRA**: `W_after = W_before + (lora_alpha / r) * lora_B @ lora_A`. Read `lora_alpha` and `r` from `adapter_config.json` in the checkpoint directory. Key mapping: base model key `model.layers.{l}.{module}.weight` maps to adapter keys `base_model.model.model.layers.{l}.{module}.lora_A.weight` and `...lora_B.weight`.
   c. Compute `ΔW = W_after - W_before`.
   d. Compute all metrics.
   e. For detailed layers: store full vector/matrix data (spectra, angles, heatmap aggregates).
   f. For all layers: store summary scalars only.
4. For detailed layers in BlockTT/SVD mode: also store factored parameter change data (metric 7, 8).
5. Write JSON output.

### Memory Management

- Process one layer at a time on CPU.
- Delete tensors after computing metrics for each module.
- Largest weight matrix is ~`[6144, 2048]` (~50MB fp32). SVD of this is feasible on CPU.
- Never load the full model into memory simultaneously.

### JSON Output Structure

```json
{
  "meta": {
    "base_model": "Qwen/Qwen3-1.7B",
    "checkpoint": "sft/blocktt_model",
    "train_mode": "blocktt",
    "top_k": 64,
    "principal_alpha": 0.1,
    "timestamp": "2026-03-16T..."
  },
  "summary": {
    "layers": [0, 1, ..., 27],
    "modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "nss": [[float, ...]],              // shape: [num_layers][num_modules]
    "max_principal_angle": [[float, ...]],  // shape: [num_layers][num_modules]
    "overlap_ratio": [[float, ...]],        // shape: [num_layers][num_modules]
    "relative_update_norm": [[float, ...]]  // shape: [num_layers][num_modules]
  },
  "detailed": {
    "layer_0.q_proj": {
      "spectrum_before": [float, ...],
      "spectrum_after": [float, ...],
      "singular_vector_angles_u": [float, ...],
      "singular_vector_angles_v": [float, ...],
      "principal_angles": [float, ...],
      "row_norms": [float, ...],
      "col_norms": [float, ...],
      "update_spectrum": [float, ...],
      "overlap_ratio": float,
      "overlap_random_baseline": float,
      "core_changes": {
        "btt_l_block_norms": [[float, ...], ...],
        "btt_r_block_norms": [[float, ...], ...]
      }
    }
  }
}
```

## `build_report.py` Specification

### CLI Interface

```
python analysis/build_report.py \
  --data <path.json> [<path2.json> ...]   # one or more analysis results
  --output <path.html>                     # default: analysis_results/report.html
```

### HTML Report Structure

Single self-contained HTML file. Plotly.js loaded from CDN. Data embedded as `<script type="application/json">` tag.

**Header bar**: run metadata (model, train mode, checkpoint, date).

**Tab 1: Summary Dashboard**
- 4 heatmap grids (rows = layers, columns = modules): NSS, max principal angle, overlap ratio, relative update norm.
- Colorscale per metric. Hover shows exact values.
- Click cell → navigates to Tab 2 for that layer+module.

**Tab 2: Detailed Layer Analysis**
- Dropdown selectors for layer and module.
- Left column:
  - Spectrum plot: overlaid `σ(W_before)` vs `σ(W_after)`, log scale Y-axis.
  - Singular vector angle plot: X = index, Y = angle (degrees), dot size = singular value magnitude.
  - Principal angles: top-k subspace angles bar chart.
- Right column:
  - Row-wise and column-wise norm bar charts.
  - Update spectrum: SVD of `ΔW` (singular values of the update).
  - Overlap visualization: overlap ratio bar (actual vs random baseline).

**Tab 3: Factored Parameters** (shown only for BlockTT/SVD)
- Per-core change heatmaps (sliced by block index for 3D tensors).
- Per-block Frobenius norm grid.

**Tab 4: Cross-layer Trends**
- Line plots across layers: NSS, max principal angle, overlap ratio.
- One line per module type. Toggle individual modules on/off.
- Grouped comparison: attention modules vs MLP modules.

### Multi-run comparison mode

When multiple JSON files are provided, the report adds:
- Side-by-side summary heatmaps.
- Overlay lines in cross-layer trend plots (different colors per run).
- Dropdown to switch between runs in detailed view.

## File Organization

```
analysis/
  analyze_weights.py
  build_report.py
  templates/
    report_template.html
analysis_results/          # gitignored
  *.json
  *.html
```

## Dependencies

- `torch`, `safetensors`, `numpy` — already in project
- `json`, `argparse`, `pathlib` — stdlib
- Plotly.js — loaded from CDN in HTML, no Python dependency

## Usage Examples

```bash
# Analyze SFT BlockTT model against base
python analysis/analyze_weights.py \
  --base-model Qwen/Qwen3-4B \
  --checkpoint sft/blocktt_model \
  --train-mode blocktt \
  --output analysis_results/blocktt_sft.json

# Analyze LoRA RL checkpoint
python analysis/analyze_weights.py \
  --base-model Qwen/Qwen3-1.7B \
  --checkpoint runs/26/step=25 \
  --train-mode lora \
  --output analysis_results/lora_rl_step25.json

# Generate single report
python analysis/build_report.py \
  --data analysis_results/blocktt_sft.json \
  --output analysis_results/blocktt_report.html

# Compare multiple runs
python analysis/build_report.py \
  --data analysis_results/blocktt_sft.json analysis_results/lora_rl_step25.json \
  --output analysis_results/comparison.html
```

## Future Extensions

- Add SVD/BlockTT RL checkpoints once `--enable-save-ckpt` runs complete.
- Temporal analysis: load multiple steps from same run, animate metrics over training.
- Add consensus ratio (requires multiple runs with same config).
