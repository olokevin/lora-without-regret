# tests/test_btt_layer_fused_flag.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
import pytest


def _make_layer(device):
    from btt_layer import BTTLayer
    d = 64
    mod = BTTLayer(d, d, rank=8).to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)
    return mod


def test_fused_flag_default_off():
    from btt_layer import BTTLayer
    assert BTTLayer.use_fused_step2 is False


def test_fused_path_matches_eager_on_cpu_fallback():
    from btt_layer import BTTLayer
    mod = _make_layer("cpu")
    x = torch.randn(8, 64)
    with torch.no_grad():
        y_eager = mod(x)
    BTTLayer.use_fused_step2 = True
    try:
        with torch.no_grad():
            y_fused = mod(x)
    finally:
        BTTLayer.use_fused_step2 = False
    assert torch.allclose(y_eager, y_fused, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fused_path_matches_eager_on_cuda():
    from btt_layer import BTTLayer
    mod = _make_layer("cuda")
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        y_eager = mod(x)
    BTTLayer.use_fused_step2 = True
    try:
        with torch.no_grad():
            y_fused = mod(x)
    finally:
        BTTLayer.use_fused_step2 = False
    assert torch.allclose(y_eager.float(), y_fused.float(), atol=1e-2, rtol=1e-2)
