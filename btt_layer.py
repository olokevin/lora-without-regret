import torch
import torch.nn as nn
from transformers.activations import ACT2FN
import numpy as np


def _closest_factor_pair(d):
    root = int(d ** 0.5)
    best_a = 1
    best_b = d
    best_diff = best_b - best_a
    for a in range(1, root + 1):
        if d % a == 0:
            b = d // a
            diff = abs(b - a)
            if diff < best_diff:
                best_a, best_b, best_diff = a, b, diff
    return best_a, best_b


@torch.no_grad()
def convert_linear_to_btt(
    model,
    btt_rank,
    decomp_mode="square",
    init_mode="default",
    forward_impl="einsum",
    skip_names=("lm_head",),
    include_names=None,
    lr_act=False,
):
    if btt_rank is None:
        btt_rank = "full"

    include_name_set = set(include_names) if include_names is not None else None
    modules_to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name in skip_names:
            continue
        if include_name_set is not None and leaf_name not in include_name_set:
            continue
        modules_to_replace.append((name, module))
    print(
        f"Converting {len(modules_to_replace)} Linear layers to BTT "
        f"(rank={btt_rank}, decomp_mode={decomp_mode}, init_mode={init_mode}, "
        f"forward_impl={forward_impl})"
    )

    for full_name, linear in modules_to_replace:
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]

        btt_layer = BTTLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=btt_rank,
            bias=(linear.bias is not None),
            lr_act=lr_act,
            decomp_mode=decomp_mode,
            init_mode=init_mode,
            forward_impl=forward_impl,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)

        btt_layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
        )

        setattr(parent, child_name, btt_layer)
    print("Finished Linear->BTT conversion")
    return [name for name, _ in modules_to_replace]


def get_blocktt_target_module_names(blocktt_type):
    if blocktt_type == "all":
        return (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    if blocktt_type == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    if blocktt_type == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    raise ValueError("blocktt_type must be one of: all, mlp, attn")


def configure_blocktt_trainability(model, train_bias=True):
    for p in model.parameters():
        p.requires_grad = False

    num_btt_layers = 0
    tuned_left_cores = 0
    tuned_right_cores = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, BTTLayer):
            continue
        num_btt_layers += 1

        if module.decomp_mode == "input_one_block":
            module.btt_l.requires_grad = True
            module.btt_r.requires_grad = False
            tuned_left_cores += 1
        elif module.decomp_mode == "output_one_block":
            module.btt_l.requires_grad = False
            module.btt_r.requires_grad = True
            tuned_right_cores += 1
        else:
            raise ValueError(
                "BlockTT PEFT expects decomp_mode to be input_one_block or output_one_block."
            )

        if module.bias is not None:
            module.bias.requires_grad = train_bias
            if train_bias:
                tuned_biases += 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_btt_layers": num_btt_layers,
        "tuned_left_cores": tuned_left_cores,
        "tuned_right_cores": tuned_right_cores,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }


class BTTLayer(nn.Module):
    """
    New BTT layout (2-core):
      ms[0]: input block dim
      ms[1]: number of input blocks
      ns[0]: number of output blocks
      ns[1]: output block dim

    Parameter shapes:
      btt_r: (ms[1], ms[0], rank * ns[0])
      btt_l: (ns[0], rank * ms[1], ns[1])
    """

    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        lr_act=True,  # actv between cores
        lr_act_type="silu",
        num_cores=2,
        decomp_mode="square", # "input_one_block", "output_one_block"
        init_mode="mup", # "default", "mup"
        forward_impl="einsum", # "bmm", "einsum"
    ):
        super(BTTLayer, self).__init__()

        if num_cores != 2:
            raise NotImplementedError("BTTLayer currently supports only c=2 cores.")

        mode_aliases = {
            "input_block": "input_one_block",
            "output_block": "output_one_block",
        }
        decomp_mode = mode_aliases.get(decomp_mode, decomp_mode)

        self.in_features = in_features
        self.out_features = out_features
        self.num_cores = num_cores
        self.decomp_mode = decomp_mode
        self.init_mode = init_mode
        if forward_impl not in {"bmm", "einsum"}:
            raise ValueError("forward_impl must be one of: bmm, einsum")
        self.forward_impl = forward_impl

        out_blocks, out_block_size = _closest_factor_pair(out_features)
        in_blocks, in_block_size = _closest_factor_pair(in_features)

        if decomp_mode == "square":
            ns_0 = out_blocks
            ns_1 = out_block_size
            ms_1 = in_blocks
            ms_0 = in_block_size
        elif decomp_mode == "output_one_block":
            ns_0 = 1
            ns_1 = out_features
            ms_1 = in_blocks
            ms_0 = in_block_size
        elif decomp_mode == "input_one_block":
            ns_0 = out_blocks
            ns_1 = out_block_size
            ms_1 = 1
            ms_0 = in_features
        else:
            raise ValueError(
                "decomp_mode must be one of: square, output_one_block, input_one_block"
            )

        self.ms = (ms_0, ms_1)
        self.ns = (ns_0, ns_1)

        # Keep old attribute names for compatibility with existing code paths.
        self.b = self.ms[0]
        self.n = self.ms[1]
        self.m = self.ns[0]
        self.a = self.ns[1]

        if isinstance(rank, str):
            if rank != "full":
                raise ValueError("rank as string must be 'full'")
            resolved_rank = min(self.ms[0], self.ns[1])
        elif isinstance(rank, float):
            if not (0 < rank < 1):
                raise ValueError("rank as float must satisfy 0 < rank < 1")
            target_params = rank * out_features * in_features
            rank_denominator = self.ns[0] * self.ms[1] * (self.ns[1] + self.ms[0])
            approx_rank = target_params / rank_denominator
            low_rank = max(1, int(np.floor(approx_rank)))
            high_rank = max(1, int(np.ceil(approx_rank)))
            low_params = self.ns[0] * self.ms[1] * low_rank * (self.ns[1] + self.ms[0])
            high_params = self.ns[0] * self.ms[1] * high_rank * (self.ns[1] + self.ms[0])
            if abs(low_params - target_params) <= abs(high_params - target_params):
                resolved_rank = low_rank
            else:
                resolved_rank = high_rank
        elif isinstance(rank, int):
            if rank <= 0:
                raise ValueError("rank as integer must be > 0")
            resolved_rank = rank
        else:
            raise TypeError("rank must be an int, a float in (0, 1), or 'full'")

        self.rank = resolved_rank

        if lr_act:
            self.lr_act = ACT2FN[lr_act_type]

        if init_mode == "default":
            target_sdv = (in_features + out_features) ** (-1 / 2)
            self.btt_r = nn.Parameter(
                torch.randn(self.ms[1], self.ms[0], self.rank * self.ns[0])
                / self.rank ** (1 / 4)
                * target_sdv ** (1 / 2)
            )
            self.btt_l = nn.Parameter(
                torch.randn(self.ns[0], self.rank * self.ms[1], self.ns[1])
                / self.rank ** (1 / 4)
                * target_sdv ** (1 / 2)
            )
        elif init_mode == "mup":
            std_r = np.sqrt(1 / self.ms[0]) * min(
                1, np.sqrt((self.rank * self.ns[0]) / self.ms[0])
            )
            std_l = np.sqrt(1 / (self.rank * self.ms[1])) * min(
                1, np.sqrt(self.ns[1] / (self.rank * self.ms[1]))
            )
            self.btt_r = nn.Parameter(
                torch.randn(self.ms[1], self.ms[0], self.rank * self.ns[0]) * std_r
            )
            self.btt_l = nn.Parameter(
                torch.randn(self.ns[0], self.rank * self.ms[1], self.ns[1]) * std_l
            )
        else:
            raise ValueError("init_mode must be one of: default, mup")

        # Metadata used by optimizer to support both old/new BTT layouts.
        self.btt_r.btt_layout = "new"
        self.btt_r.btt_rank = self.rank
        self.btt_r.btt_ms0 = self.ms[0]
        self.btt_r.btt_ms1 = self.ms[1]
        self.btt_r.btt_ns0 = self.ns[0]
        self.btt_r.btt_ns1 = self.ns[1]

        self.btt_l.btt_layout = "new"
        self.btt_l.btt_rank = self.rank
        self.btt_l.btt_ms0 = self.ms[0]
        self.btt_l.btt_ms1 = self.ms[1]
        self.btt_l.btt_ns0 = self.ns[0]
        self.btt_l.btt_ns1 = self.ns[1]

        if bias == False:
            self.register_parameter("bias", None)
        else:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return (
            f"mode: {self.decomp_mode}, init: {self.init_mode}, "
            f"forward: {self.forward_impl}, "
            f"ms: ({self.ms[0]}, {self.ms[1]}), "
            f"ns: ({self.ns[0]}, {self.ns[1]}), "
            f"rank: {self.rank}, "
            f"btt_r: {self.btt_r.shape}, "
            f"btt_l: {self.btt_l.shape}, "
            f"bias: {self.bias.shape if self.bias is not None else False}"
        )

    @torch.no_grad()
    def init_from_linear_weight(self, weight, bias=None):
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Linear weight shape must be {(self.out_features, self.in_features)}, "
                f"got {tuple(weight.shape)}"
            )
        if weight.shape != (self.ns[0] * self.ns[1], self.ms[1] * self.ms[0]):
            raise ValueError(
                f"Linear weight shape {tuple(weight.shape)} not compatible with "
                f"BTT blocks (ns={self.ns}, ms={self.ms})"
            )

        param_dtype = weight.dtype
        blocks = weight.reshape(self.ns[0], self.ns[1], self.ms[1], self.ms[0])
        blocks = blocks.permute(0, 2, 1, 3).reshape(
            self.ns[0] * self.ms[1], self.ns[1], self.ms[0]
        )
        svd_dtype = (
            torch.float32
            if param_dtype in (torch.float16, torch.bfloat16)
            else param_dtype
        )
        U, S, Vh = torch.linalg.svd(blocks.to(dtype=svd_dtype), full_matrices=False)

        max_svd_rank = min(self.ns[1], self.ms[0])
        use_rank = min(self.rank, max_svd_rank)

        core_l = torch.zeros(
            self.ns[0] * self.ms[1],
            self.ns[1],
            self.rank,
            device=weight.device,
            dtype=param_dtype,
        )
        core_r = torch.zeros(
            self.ns[0] * self.ms[1],
            self.rank,
            self.ms[0],
            device=weight.device,
            dtype=param_dtype,
        )

        sqrt_s = torch.sqrt(S[:, :use_rank]).to(dtype=param_dtype)
        core_l[:, :, :use_rank] = (
            U[:, :, :use_rank].to(dtype=param_dtype) * sqrt_s.unsqueeze(1)
        )
        core_r[:, :use_rank, :] = (
            sqrt_s.unsqueeze(-1) * Vh[:, :use_rank, :].to(dtype=param_dtype)
        )

        core_l = core_l.reshape(self.ns[0], self.ms[1], self.ns[1], self.rank)
        core_r = core_r.reshape(self.ns[0], self.ms[1], self.rank, self.ms[0])
        packed_l = core_l.permute(0, 1, 3, 2).reshape(
            self.ns[0], self.rank * self.ms[1], self.ns[1]
        )
        packed_r = core_r.permute(1, 3, 0, 2).reshape(
            self.ms[1], self.ms[0], self.rank * self.ns[0]
        )

        self.btt_l.data.copy_(packed_l)
        self.btt_r.data.copy_(packed_r)

        if bias is not None:
            if self.bias is None:
                raise ValueError("BTTLayer has no bias parameter but bias tensor was provided")
            self.bias.data.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        elif self.bias is not None:
            raise ValueError(
                "BTTLayer has bias parameter but no source bias was provided. "
                "Construct BTTLayer with bias=False for biasless source layers."
            )

    def _forward_bmm(self, x, orig_shape):
        batch_n = x.shape[0]
        # Step 1: (n, B, b) @ (n, b, m*r) -> (n, B, m*r)
        inner = torch.bmm(x.transpose(0, 1).contiguous(), self.btt_r)
        inner = inner.reshape(self.ms[1], batch_n, self.ns[0], self.rank)
        inner = inner.permute(2, 1, 0, 3).contiguous()  # (m, B, n, r)

        if hasattr(self, "lr_act"):
            inner = self.lr_act(inner)

        # Step 2: (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        out = torch.bmm(
            inner.reshape(self.ns[0], batch_n, self.rank * self.ms[1]),
            self.btt_l,
        )
        out = out.permute(1, 0, 2).contiguous().reshape(
            *orig_shape[:-1], self.out_features
        )

        if self.bias is not None:
            out += self.bias

        return out

    def _forward_einsum(self, x, orig_shape):
        # Step 1: (B, n, b) @ (n, b, m*r) -> (B, n, m*r)
        inner = torch.einsum("Bnb,nbk->Bnk", x, self.btt_r)
        inner = inner.reshape(x.shape[0], self.ms[1], self.ns[0], self.rank)  # (B, n, m, r)

        if hasattr(self, "lr_act"):
            inner = self.lr_act(inner)

        # Step 2: (B, n, m, r) x (m, n, r, a) -> (B, m, a)
        btt_l = self.btt_l.reshape(self.ns[0], self.ms[1], self.rank, self.ns[1])
        out = torch.einsum("Bnmr,mnra->Bma", inner, btt_l)
        out = out.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out += self.bias

        return out

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"BTTLayer expected last dim {self.in_features}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x = x.reshape(-1, self.ms[1], self.ms[0])  # (B, n, b)

        if self.forward_impl == "bmm":
            return self._forward_bmm(x, orig_shape)
        if self.forward_impl == "einsum":
            return self._forward_einsum(x, orig_shape)

        raise RuntimeError(f"Unknown forward_impl: {self.forward_impl}")
