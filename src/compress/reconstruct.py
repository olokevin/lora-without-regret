"""Reconstruct BTT/SVD compressed state dicts into standard nn.Linear weight tensors."""

import torch


def _reconstruct_btt(sd: dict, prefix: str) -> dict:
    """Reconstruct nn.Linear weight from BTTLinear parameters.

    BTTLinear forward:
        inner = x @ btt_r[0]                          # (B, m*max_rank)
        inner = inner.reshape(B, m, max_rank)
        out   = bmm(inner.permute(1,0,2), btt_l)      # (m, B, a)
        out   = out.permute(1,0,2).reshape(B, m*a)
        if inv_permutation: out = out[:, inv_permutation]
        if bias: out = out + bias

    Reconstruction:
        For each block i:  block_i = btt_r[0][:, i*max_rank:(i+1)*max_rank] @ btt_l[i]
                                     shape (d_in, a)
        W_T = cat(blocks, dim=1)     shape (d_in, d_out)
        if inv_permutation: W_T = W_T[:, inv_permutation]
        W   = W_T.T                  shape (d_out, d_in)
    """
    btt_l = sd[f"{prefix}.btt_l"].float()       # (m, max_rank, a)
    btt_r = sd[f"{prefix}.btt_r"].float()       # (1, d_in, m*max_rank)
    m, max_rank, _ = btt_l.shape

    r0 = btt_r[0]  # (d_in, m*max_rank)
    blocks = [r0[:, i * max_rank:(i + 1) * max_rank] @ btt_l[i] for i in range(m)]
    W_T = torch.cat(blocks, dim=1)              # (d_in, d_out)

    inv_perm_key = f"{prefix}.inv_permutation"
    if inv_perm_key in sd:
        W_T = W_T[:, sd[inv_perm_key].long()]

    out = {f"{prefix}.weight": W_T.T.contiguous()}
    if f"{prefix}.bias" in sd:
        out[f"{prefix}.bias"] = sd[f"{prefix}.bias"]
    return out


def _reconstruct_svd(sd: dict, prefix: str) -> dict:
    """Reconstruct nn.Linear weight from SVDCompressedLinear parameters.

    SVDCompressedLinear forward:
        out = x @ V_r @ U_r + bias

    Reconstruction:
        W = (V_r @ U_r).T   shape (d_out, d_in)
    """
    V_r = sd[f"{prefix}.V_r"].float()   # (d_in, rank)
    U_r = sd[f"{prefix}.U_r"].float()   # (rank, d_out)
    W = (V_r @ U_r).T.contiguous()     # (d_out, d_in)

    out = {f"{prefix}.weight": W}
    if f"{prefix}.bias" in sd:
        out[f"{prefix}.bias"] = sd[f"{prefix}.bias"]
    return out


def reconstruct_state_dict(sd: dict) -> dict:
    """Convert a compressed state dict to a standard one with nn.Linear weights.

    Handles BTTLinear (keys: btt_l, btt_r) and SVDCompressedLinear (keys: V_r, U_r).
    All other keys are passed through unchanged.
    Result tensors are cast back to the original dtype (bfloat16 if input was float32).
    """
    btt_prefixes = {k[: -len(".btt_l")] for k in sd if k.endswith(".btt_l")}
    svd_prefixes = {k[: -len(".V_r")] for k in sd if k.endswith(".V_r")}

    compressed_keys: set = set()
    for p in btt_prefixes:
        compressed_keys.update(
            k for k in sd if k.startswith(p + ".") and
            k[len(p) + 1:] in ("btt_l", "btt_r", "bias", "inv_permutation")
        )
    for p in svd_prefixes:
        compressed_keys.update(
            k for k in sd if k.startswith(p + ".") and
            k[len(p) + 1:] in ("V_r", "U_r", "bias")
        )

    new_sd: dict = {}
    for k, v in sd.items():
        if k not in compressed_keys:
            new_sd[k] = v

    for prefix in sorted(btt_prefixes):
        new_sd.update(_reconstruct_btt(sd, prefix))

    for prefix in sorted(svd_prefixes):
        new_sd.update(_reconstruct_svd(sd, prefix))

    # Cast reconstructed float32 tensors back to bfloat16
    for k in new_sd:
        if new_sd[k].dtype == torch.float32:
            new_sd[k] = new_sd[k].to(torch.bfloat16)

    return new_sd
