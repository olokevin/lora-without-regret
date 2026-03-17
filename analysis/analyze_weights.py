"""Weight update analysis: compute metrics comparing base model vs trained checkpoint."""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime
from safetensors import safe_open


def materialize_blocktt_weight(
    btt_l: torch.Tensor, btt_r: torch.Tensor, btt_s: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct dense weight from BlockTT cores.

    Args:
        btt_l: (m, rank*n, a) — left/output core
        btt_r: (n, b, m*rank) — right/input core
        btt_s: (m, n, rank) — optional singular values

    Returns:
        Dense weight tensor of shape (m*a, n*b).
    """
    m, _, a = btt_l.shape
    n, b, _ = btt_r.shape
    rank = btt_r.shape[2] // m

    # Reshape to (m, n, rank, a) and (m, n, rank, b)
    l = btt_l.reshape(m, n, rank, a)
    r = btt_r.reshape(n, b, m, rank).permute(2, 0, 3, 1)  # (m, n, rank, b)

    if btt_s is not None:
        l = l * btt_s.unsqueeze(-1)  # (m, n, rank, a) * (m, n, rank, 1)

    # Contract over rank: (m, n, a, b)
    w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)

    # Assemble into (m*a, n*b)
    return w_blocks.permute(0, 2, 1, 3).reshape(m * a, n * b)


def materialize_svd_weight(
    svd_a: torch.Tensor, svd_b: torch.Tensor, svd_s: torch.Tensor | None = None
) -> torch.Tensor:
    """Reconstruct dense weight from SVD factors.

    Args:
        svd_a: (out_features, rank)
        svd_b: (rank, in_features)
        svd_s: (rank,) — optional singular values

    Returns:
        Dense weight of shape (out_features, in_features).
    """
    if svd_s is not None:
        return (svd_a * svd_s.unsqueeze(0)) @ svd_b
    return svd_a @ svd_b


def reconstruct_lora_weight(
    W_base: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_alpha: float,
    r: int,
) -> torch.Tensor:
    """Reconstruct dense weight from base weight + LoRA adapter.

    Args:
        W_base: (out_features, in_features) — original linear weight
        lora_A: (rank, in_features)
        lora_B: (out_features, rank)
        lora_alpha: LoRA scaling alpha
        r: LoRA rank

    Returns:
        W_base + (lora_alpha / r) * lora_B @ lora_A
    """
    scaling = lora_alpha / r
    return W_base + scaling * (lora_B @ lora_A)


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")


def _module_prefix(layer_idx: int, module_name: str) -> str:
    if module_name in ATTN_MODULES:
        return f"model.layers.{layer_idx}.self_attn.{module_name}"
    return f"model.layers.{layer_idx}.mlp.{module_name}"


def get_base_weight_key(layer_idx: int, module_name: str) -> str:
    return f"{_module_prefix(layer_idx, module_name)}.weight"


def get_checkpoint_keys(
    layer_idx: int, module_name: str, train_mode: str
) -> dict[str, str]:
    """Return dict mapping role -> safetensors key for a given module."""
    prefix = _module_prefix(layer_idx, module_name)
    if train_mode == "blocktt":
        return {
            "btt_l": f"{prefix}.btt_l",
            "btt_r": f"{prefix}.btt_r",
            "btt_s": f"{prefix}.btt_s",
        }
    elif train_mode == "svd":
        return {
            "svd_a": f"{prefix}.svd_a",
            "svd_b": f"{prefix}.svd_b",
            "svd_s": f"{prefix}.svd_s",
        }
    elif train_mode == "lora":
        lora_prefix = f"base_model.model.{prefix}"
        return {
            "lora_A": f"{lora_prefix}.lora_A.weight",
            "lora_B": f"{lora_prefix}.lora_B.weight",
        }
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")


def compute_update_row_col_norms(
    delta_W: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise and column-wise L2 norms of the update matrix.

    Returns:
        (row_norms, col_norms) as numpy arrays.
    """
    row_norms = torch.norm(delta_W, dim=1).numpy()
    col_norms = torch.norm(delta_W, dim=0).numpy()
    return row_norms, col_norms


def compute_singular_vector_angles(
    W_before: torch.Tensor, W_after: torch.Tensor, top_k: int
) -> tuple[list[float], list[float]]:
    """Angle (degrees) between each top-k singular vector before/after.

    Uses absolute inner product to handle SVD sign ambiguity.
    """
    U0, _, Vt0 = torch.linalg.svd(W_before.float(), full_matrices=False)
    U1, _, Vt1 = torch.linalg.svd(W_after.float(), full_matrices=False)
    k = min(top_k, U0.shape[1], U1.shape[1])

    angles_u = []
    angles_v = []
    for i in range(k):
        cos_u = torch.clamp(torch.abs(U0[:, i] @ U1[:, i]), 0.0, 1.0)
        angles_u.append(float(torch.acos(cos_u).rad2deg()))
        cos_v = torch.clamp(torch.abs(Vt0[i] @ Vt1[i]), 0.0, 1.0)
        angles_v.append(float(torch.acos(cos_v).rad2deg()))
    return angles_u, angles_v


def compute_spectrum_and_nss(
    W_before: torch.Tensor, W_after: torch.Tensor
) -> tuple[list[float], list[float], float]:
    """Singular value spectra and normalized spectral shift.

    NSS = ||sigma(W_after) - sigma(W_before)||_2 / ||sigma(W_before)||_2
    """
    s_before = torch.linalg.svdvals(W_before.float())
    s_after = torch.linalg.svdvals(W_after.float())
    norm_before = torch.norm(s_before)
    nss = float(torch.norm(s_after - s_before) / norm_before) if norm_before > 0 else 0.0
    return s_before.tolist(), s_after.tolist(), nss


def compute_principal_angles(
    W_before: torch.Tensor, W_after: torch.Tensor, top_k: int
) -> list[float]:
    """Top-k subspace principal angles (degrees).

    cos theta_i = sigma_i(U_before_k^T @ U_after_k)
    """
    U0, _, _ = torch.linalg.svd(W_before.float(), full_matrices=False)
    U1, _, _ = torch.linalg.svd(W_after.float(), full_matrices=False)
    k = min(top_k, U0.shape[1], U1.shape[1])
    cos_angles = torch.linalg.svdvals(U0[:, :k].T @ U1[:, :k])
    cos_angles = torch.clamp(cos_angles, 0.0, 1.0)
    angles_deg = torch.acos(cos_angles).rad2deg()
    return angles_deg.tolist()


def compute_principal_weight_overlap(
    W_before: torch.Tensor,
    delta_W: torch.Tensor,
    top_k: int,
    alpha: float,
    threshold_frac: float,
) -> tuple[float, float]:
    """Overlap between update mask and principal weight mask.

    Principal mask: top-alpha fraction of |rank-k SVD reconstruction| by magnitude.
    Update mask: |delta_W_ij| > threshold_frac * max(|delta_W|).

    Returns:
        (overlap_ratio, random_baseline=alpha)
    """
    # Principal weight mask from rank-k approximation
    U, S, Vt = torch.linalg.svd(W_before.float(), full_matrices=False)
    k = min(top_k, len(S))
    W_lowrank = (U[:, :k] * S[:k].unsqueeze(0)) @ Vt[:k]
    n_principal = max(1, int(alpha * W_lowrank.numel()))
    threshold_princ = torch.topk(W_lowrank.abs().flatten(), n_principal).values[-1]
    M_princ = W_lowrank.abs() >= threshold_princ

    # Update mask
    max_delta = delta_W.abs().max()
    if max_delta == 0:
        return 0.0, alpha
    M_update = delta_W.abs() > threshold_frac * max_delta

    n_update = M_update.sum().item()
    if n_update == 0:
        return 0.0, alpha

    overlap = float((M_princ & M_update).sum().item() / n_update)
    return overlap, alpha


def compute_update_spectrum(delta_W: torch.Tensor) -> list[float]:
    """Singular values of the update matrix itself."""
    s = torch.linalg.svdvals(delta_W.float())
    return s.tolist()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Analyze weight updates in trained checkpoints")
    p.add_argument("--base-model", required=True, help="Path or HF ID of base model")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    p.add_argument("--train-mode", required=True, choices=["blocktt", "svd", "lora"])
    p.add_argument("--output", default="analysis_results/metrics.json")
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--principal-alpha", type=float, default=0.1)
    p.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer indices for detailed analysis (default: 0,mid,last)")
    p.add_argument("--update-threshold", type=float, default=0.01)
    args = p.parse_args(argv)
    if args.layers is not None:
        args.layers = [int(x) for x in args.layers.split(",")]
    return args


def load_safetensors_index(directory: str) -> dict[str, str]:
    """Map tensor keys to their safetensors file paths."""
    directory = Path(directory)
    index = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                index[key] = str(f)
    return index


def load_tensor(index: dict[str, str], key: str) -> torch.Tensor | None:
    """Load a single tensor by key, or return None if key not found."""
    if key not in index:
        return None
    with safe_open(index[key], framework="pt") as sf:
        return sf.get_tensor(key)


def detect_num_layers(index: dict[str, str]) -> int:
    """Detect number of transformer layers from key names."""
    layers = set()
    for key in index:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layers.add(int(parts[i + 1]))
    return max(layers) + 1 if layers else 0


def analyze(args) -> dict:
    """Main analysis: load weights, compute metrics, return results dict."""
    base_index = load_safetensors_index(args.base_model)

    # If base_model is an HF model ID, resolve to cache path
    if not base_index:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(args.base_model)
        base_index = load_safetensors_index(local_path)

    ckpt_index = load_safetensors_index(args.checkpoint)

    num_layers = detect_num_layers(base_index)
    if args.layers is None:
        args.layers = [0, num_layers // 2, num_layers - 1]
    for l in args.layers:
        if l >= num_layers:
            raise ValueError(f"Layer {l} out of range (model has {num_layers} layers)")

    # Read LoRA config if needed
    lora_alpha, lora_r = 1.0, 1
    if args.train_mode == "lora":
        config_path = Path(args.checkpoint) / "adapter_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        lora_alpha = cfg["lora_alpha"]
        lora_r = cfg["r"]

    detailed_layers = set(args.layers)

    results = {
        "meta": {
            "base_model": args.base_model,
            "checkpoint": args.checkpoint,
            "train_mode": args.train_mode,
            "top_k": args.top_k,
            "principal_alpha": args.principal_alpha,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "layers": list(range(num_layers)),
            "modules": list(TARGET_MODULES),
            "nss": [],
            "max_principal_angle": [],
            "overlap_ratio": [],
            "relative_update_norm": [],
        },
        "detailed": {},
    }

    for layer_idx in range(num_layers):
        print(f"Processing layer {layer_idx}/{num_layers - 1}...")
        layer_nss = []
        layer_angles = []
        layer_overlap = []
        layer_norms = []

        for mod in TARGET_MODULES:
            # Load W_before
            base_key = get_base_weight_key(layer_idx, mod)
            W_before = load_tensor(base_index, base_key)
            if W_before is None:
                layer_nss.append(0.0)
                layer_angles.append(0.0)
                layer_overlap.append(0.0)
                layer_norms.append(0.0)
                continue

            W_before = W_before.float()

            # Reconstruct W_after
            ckpt_keys = get_checkpoint_keys(layer_idx, mod, args.train_mode)
            if args.train_mode == "blocktt":
                btt_l = load_tensor(ckpt_index, ckpt_keys["btt_l"])
                btt_r = load_tensor(ckpt_index, ckpt_keys["btt_r"])
                btt_s = load_tensor(ckpt_index, ckpt_keys["btt_s"])
                if btt_l is None or btt_r is None:
                    layer_nss.append(0.0)
                    layer_angles.append(0.0)
                    layer_overlap.append(0.0)
                    layer_norms.append(0.0)
                    continue
                W_after = materialize_blocktt_weight(btt_l.float(), btt_r.float(), btt_s=btt_s.float() if btt_s is not None else None)
            elif args.train_mode == "svd":
                svd_a = load_tensor(ckpt_index, ckpt_keys["svd_a"])
                svd_b = load_tensor(ckpt_index, ckpt_keys["svd_b"])
                svd_s = load_tensor(ckpt_index, ckpt_keys["svd_s"])
                if svd_a is None or svd_b is None:
                    layer_nss.append(0.0)
                    layer_angles.append(0.0)
                    layer_overlap.append(0.0)
                    layer_norms.append(0.0)
                    continue
                W_after = materialize_svd_weight(svd_a.float(), svd_b.float(), svd_s=svd_s.float() if svd_s is not None else None)
            elif args.train_mode == "lora":
                lora_A = load_tensor(ckpt_index, ckpt_keys["lora_A"])
                lora_B = load_tensor(ckpt_index, ckpt_keys["lora_B"])
                if lora_A is None or lora_B is None:
                    layer_nss.append(0.0)
                    layer_angles.append(0.0)
                    layer_overlap.append(0.0)
                    layer_norms.append(0.0)
                    continue
                W_after = reconstruct_lora_weight(W_before, lora_A.float(), lora_B.float(), lora_alpha, lora_r)

            delta_W = W_after - W_before

            # Compute spectra once (reuse for summary + detail)
            s_before, s_after, nss = compute_spectrum_and_nss(W_before, W_after)
            pa = compute_principal_angles(W_before, W_after, args.top_k)
            overlap, baseline = compute_principal_weight_overlap(
                W_before, delta_W, args.top_k, args.principal_alpha, args.update_threshold
            )
            rel_norm = float(torch.norm(delta_W) / torch.norm(W_before)) if torch.norm(W_before) > 0 else 0.0

            layer_nss.append(nss)
            layer_angles.append(max(pa) if pa else 0.0)
            layer_overlap.append(overlap)
            layer_norms.append(rel_norm)

            # Detailed metrics for selected layers
            if layer_idx in detailed_layers:
                angles_u, angles_v = compute_singular_vector_angles(W_before, W_after, args.top_k)
                row_norms, col_norms = compute_update_row_col_norms(delta_W)
                update_s = compute_update_spectrum(delta_W)

                detail_key = f"layer_{layer_idx}.{mod}"
                results["detailed"][detail_key] = {
                    "spectrum_before": s_before,
                    "spectrum_after": s_after,
                    "singular_vector_angles_u": angles_u,
                    "singular_vector_angles_v": angles_v,
                    "principal_angles": pa,
                    "row_norms": row_norms.tolist(),
                    "col_norms": col_norms.tolist(),
                    "update_spectrum": update_s,
                    "overlap_ratio": overlap,
                    "overlap_random_baseline": baseline,
                }

            # Free memory
            del W_before, W_after, delta_W

        results["summary"]["nss"].append(layer_nss)
        results["summary"]["max_principal_angle"].append(layer_angles)
        results["summary"]["overlap_ratio"].append(layer_overlap)
        results["summary"]["relative_update_norm"].append(layer_norms)

    return results


def main():
    args = parse_args()
    results = analyze(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
