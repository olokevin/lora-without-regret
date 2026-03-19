import torch
import torch.nn as nn
from transformers.activations import ACT2FN
import numpy as np
import warnings
import ast
import json
import re


VALID_S_MERGED_TO = {"frozen", "trainable", "output", "input", "split", "keep"}
VALID_BLOCKTT_DECOMP_MODES = {"square", "input_one_block", "output_one_block"}
BLOCKTT_DECOMP_MODE_ALIASES = {
    "input": "input_one_block",
    "output": "output_one_block",
    "input_block": "input_one_block",
    "output_block": "output_one_block",
}
BLOCKTT_DECOMP_GROUP_TO_MODULES = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "o": ("o_proj",),
    "mlp_upgate": ("gate_proj", "up_proj"),
    "mlp_down": ("down_proj",),
}
BLOCKTT_DECOMP_GROUP_ALIASES = {
    "mlp_up_gate": "mlp_upgate",
    "upgate": "mlp_upgate",
    "up_gate": "mlp_upgate",
}
_BARE_DICT_KEY_PATTERN = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:')
_BARE_MODE_VALUE_PATTERN = re.compile(
    r'(:\s*)(input_one_block|output_one_block|input|output|input_block|output_block|square)\s*([,}])'
)


def normalize_blocktt_decomp_mode(mode, allow_square=True):
    if not isinstance(mode, str):
        raise ValueError("BlockTT decomp mode must be a string")
    canonical_mode = BLOCKTT_DECOMP_MODE_ALIASES.get(mode.strip(), mode.strip())
    valid_modes = VALID_BLOCKTT_DECOMP_MODES if allow_square else {
        "input_one_block",
        "output_one_block",
    }
    if canonical_mode not in valid_modes:
        allowed = ", ".join(sorted(valid_modes))
        raise ValueError(f"BlockTT decomp mode must be one of: {allowed}")
    return canonical_mode


def _parse_decomp_mode_mapping_literal(raw_value):
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    stripped = raw_value.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None

    parse_attempts = []
    parse_attempts.append(stripped)
    sanitized = _BARE_DICT_KEY_PATTERN.sub(r'\1"\2":', stripped)
    sanitized = _BARE_MODE_VALUE_PATTERN.sub(r'\1"\2"\3', sanitized)
    if sanitized != stripped:
        parse_attempts.append(sanitized)

    last_exc = None
    for candidate in parse_attempts:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except (ValueError, SyntaxError, TypeError) as exc:
                last_exc = exc
                continue
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("--decomp-mode dict literal must parse to a dictionary")

    raise ValueError(
        "--decomp-mode dictionary could not be parsed. "
        "Use JSON/Python dict syntax, e.g. "
        '\'{"qkv":"input","o":"output","mlp_upgate":"output","mlp_down":"output"}\''
    ) from last_exc


def resolve_blocktt_decomp_modes(decomp_mode, include_names=None, default_mode="input_one_block"):
    include_name_set = set(include_names) if include_names is not None else None
    default_canonical = normalize_blocktt_decomp_mode(default_mode, allow_square=False)

    parsed_mapping = _parse_decomp_mode_mapping_literal(decomp_mode)
    if parsed_mapping is None:
        scalar_mode = normalize_blocktt_decomp_mode(decomp_mode, allow_square=False)
        if include_name_set is None:
            module_modes = {}
        else:
            module_modes = {module_name: scalar_mode for module_name in include_name_set}
        return scalar_mode, module_modes

    group_modes = {}
    for raw_key, raw_value in parsed_mapping.items():
        if not isinstance(raw_key, str):
            raise ValueError("--decomp-mode dict keys must be strings")
        normalized_key = BLOCKTT_DECOMP_GROUP_ALIASES.get(raw_key.strip(), raw_key.strip())
        if normalized_key not in BLOCKTT_DECOMP_GROUP_TO_MODULES:
            valid_keys = ", ".join(sorted(BLOCKTT_DECOMP_GROUP_TO_MODULES))
            raise ValueError(
                f"Invalid --decomp-mode key '{raw_key}'. Expected one of: {valid_keys}"
            )
        group_modes[normalized_key] = normalize_blocktt_decomp_mode(
            str(raw_value), allow_square=False
        )

    module_modes = {}
    for group_name, module_names in BLOCKTT_DECOMP_GROUP_TO_MODULES.items():
        group_mode = group_modes.get(group_name, default_canonical)
        for module_name in module_names:
            module_modes[module_name] = group_mode

    if include_name_set is not None:
        missing_names = include_name_set - set(module_modes)
        if missing_names:
            raise ValueError(
                f"Cannot resolve decomp modes for modules: {sorted(missing_names)}"
            )
        module_modes = {
            module_name: module_modes[module_name]
            for module_name in include_name_set
        }

    canonical_groups = {
        group_name: group_modes.get(group_name, default_canonical)
        for group_name in BLOCKTT_DECOMP_GROUP_TO_MODULES
    }
    return canonical_groups, module_modes


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


def _resolve_blocktt_trainable_sides(left_size, right_size, train_position):
    if train_position not in {"small", "large", "both"}:
        raise ValueError("BlockTT train_position must be one of: small, large, both")

    if train_position == "both":
        return True, True
    if train_position == "small":
        # Tie-break to left core for deterministic behavior.
        train_left = left_size <= right_size
        return train_left, not train_left

    # Tie-break to left core for deterministic behavior.
    train_left = left_size >= right_size
    return train_left, not train_left


def resolve_blocktt_s_merged_to(train_position, s_merged_to=None, left_size=None, right_size=None):
    if s_merged_to is None:
        if train_position == "both":
            return "split"
        s_merged_to = "frozen"

    if s_merged_to not in VALID_S_MERGED_TO:
        raise ValueError(
            "s_merged_to must be one of: frozen, trainable, output, input, split, keep"
        )

    if s_merged_to in {"output", "input", "split", "keep"}:
        return s_merged_to

    if left_size is None or right_size is None:
        raise ValueError("left_size and right_size are required for frozen/trainable aliases")

    train_left, train_right = _resolve_blocktt_trainable_sides(
        left_size=left_size,
        right_size=right_size,
        train_position=train_position,
    )
    if train_left and train_right:
        raise ValueError(
            "BlockTT s_merged_to frozen/trainable is invalid when both cores are trainable. "
            "Use output, input, or split."
        )

    if s_merged_to == "trainable":
        return "output" if train_left else "input"
    return "input" if train_left else "output"


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
    s_merged_to=None,
    train_position="small",
):
    if btt_rank is None:
        btt_rank = "full"
    if forward_impl != "einsum":
        warnings.warn(
            "forward_impl is ignored by the canonical BTTLayer implementation.",
            stacklevel=2,
        )

    include_name_set = set(include_names) if include_names is not None else None
    if isinstance(decomp_mode, dict):
        normalized_decomp_mode = {
            str(name): normalize_blocktt_decomp_mode(mode, allow_square=False)
            for name, mode in decomp_mode.items()
        }
        default_decomp_mode = None
        decomp_mode_printable = normalized_decomp_mode
    else:
        normalized_decomp_mode = None
        default_decomp_mode = normalize_blocktt_decomp_mode(decomp_mode)
        decomp_mode_printable = default_decomp_mode

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
        f"(rank={btt_rank}, decomp_mode={decomp_mode_printable}, init_mode={init_mode}, "
        f"forward_impl={forward_impl})"
    )

    for full_name, linear in modules_to_replace:
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]
        layer_decomp_mode = default_decomp_mode
        if normalized_decomp_mode is not None:
            if child_name not in normalized_decomp_mode:
                raise ValueError(
                    f"Missing per-module decomp mode for '{child_name}' in --decomp-mode dict"
                )
            layer_decomp_mode = normalized_decomp_mode[child_name]

        btt_layer = BTTLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=btt_rank,
            bias=(linear.bias is not None),
            lr_act=lr_act,
            decomp_mode=layer_decomp_mode,
            init_mode=init_mode,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)

        btt_layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
            s_merged_to=s_merged_to,
            train_position=train_position,
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


def configure_blocktt_trainability(model, train_bias=True, train_position="small"):
    if train_position not in {"small", "large", "both"}:
        raise ValueError("BlockTT train_position must be one of: small, large, both")

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

        left_size = module.btt_l.numel()
        right_size = module.btt_r.numel()

        train_left, train_right = _resolve_blocktt_trainable_sides(
            left_size=left_size,
            right_size=right_size,
            train_position=train_position,
        )

        module.btt_l.requires_grad = train_left
        module.btt_r.requires_grad = train_right
        if module.btt_s is not None:
            module.btt_s.requires_grad = False
        tuned_left_cores += int(train_left)
        tuned_right_cores += int(train_right)

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


@torch.no_grad()
def normalize_trainable_blocktt_cores_(model, eps=1e-12):
    normalized_left = 0
    normalized_right = 0
    for module in model.modules():
        if not isinstance(module, BTTLayer):
            continue

        if module.btt_r.requires_grad:
            # Packed shape (n, b, m*r): each fixed (n, m, r) vector over b should be unit norm.
            norms = torch.linalg.vector_norm(module.btt_r, dim=1, keepdim=True).clamp_min(eps)
            module.btt_r.div_(norms)
            normalized_right += 1

        if module.btt_l.requires_grad:
            # Packed shape (m, n*r, a): each fixed (m, n, r) vector over a should be unit norm.
            norms = torch.linalg.vector_norm(module.btt_l, dim=2, keepdim=True).clamp_min(eps)
            module.btt_l.div_(norms)
            normalized_left += 1

    return {
        "normalized_left_cores": normalized_left,
        "normalized_right_cores": normalized_right,
    }



class BTTLayer(nn.Module):
    """
    New BTT layout (2-core) using canonical dimensions:
      b: input block dim
      n: number of input blocks
      m: number of output blocks
      a: output block dim

    Parameter shapes:
      btt_r: (n, b, m * rank)
      btt_l: (m, rank * n, a)
    """

    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        lr_act=True,
        lr_act_type="silu",
        num_cores=2,
        decomp_mode="square",
        init_mode="default",
    ):
        super(BTTLayer, self).__init__()

        if num_cores != 2:
            raise NotImplementedError("BTTLayer currently supports only c=2 cores.")

        decomp_mode = normalize_blocktt_decomp_mode(decomp_mode)

        self.in_features = in_features
        self.out_features = out_features
        self.num_cores = num_cores
        self.decomp_mode = decomp_mode
        self.init_mode = init_mode
        self.lr_act = lr_act
        self.lr_act_type = lr_act_type
        self.use_gate_proj = lr_act and lr_act_type == "swiglu"

        out_blocks, out_block_size = _closest_factor_pair(out_features)
        in_blocks, in_block_size = _closest_factor_pair(in_features)

        if decomp_mode == "square":
            m = out_blocks
            a = out_block_size
            n = in_blocks
            b = in_block_size
        elif decomp_mode == "output_one_block":
            m = 1
            a = out_features
            n = in_blocks
            b = in_block_size
        elif decomp_mode == "input_one_block":
            m = out_blocks
            a = out_block_size
            n = 1
            b = in_features
        else:
            raise ValueError(
                "decomp_mode must be one of: square, output_one_block, input_one_block"
            )

        self.b = b
        self.n = n
        self.m = m
        self.a = a

        if isinstance(rank, str):
            if rank != "full":
                raise ValueError("rank as string must be 'full'")
            resolved_rank = min(self.b, self.a)
        elif isinstance(rank, float):
            if not (0 < rank < 1):
                raise ValueError("rank as float must satisfy 0 < rank < 1")
            target_params = rank * out_features * in_features
            rank_denominator = self.m * self.n * (self.a + self.b)
            approx_rank = target_params / rank_denominator
            low_rank = max(1, int(np.floor(approx_rank)))
            high_rank = max(1, int(np.ceil(approx_rank)))
            low_params = self.m * self.n * low_rank * (self.a + self.b)
            high_params = self.m * self.n * high_rank * (self.a + self.b)
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

        if lr_act and (not self.use_gate_proj):
            self.act_fn = ACT2FN[lr_act_type]

        if init_mode == "default":
            target_sdv = (in_features + out_features) ** (-1 / 2)
            self.btt_r = nn.Parameter(
                torch.randn(self.n, self.b, self.m * self.rank)
                / self.rank ** (1 / 4)
                * target_sdv ** (1 / 2)
            )
            if self.use_gate_proj:
                self.btt_g = nn.Parameter(
                    torch.randn(self.n, self.b, self.m * self.rank)
                    / self.rank ** (1 / 4)
                    * target_sdv ** (1 / 2)
                )
            self.btt_l = nn.Parameter(
                torch.randn(self.m, self.rank * self.n, self.a)
                / self.rank ** (1 / 4)
                * target_sdv ** (1 / 2)
            )
        elif init_mode == "mup":
            std_r = np.sqrt(1 / self.b) * min(
                1, np.sqrt((self.m * self.rank) / self.b)
            )
            std_l = np.sqrt(1 / (self.rank * self.n)) * min(
                1, np.sqrt(self.a / (self.rank * self.n))
            )
            self.btt_r = nn.Parameter(
                torch.randn(self.n, self.b, self.m * self.rank) * std_r
            )
            if self.use_gate_proj:
                self.btt_g = nn.Parameter(
                    torch.randn(self.n, self.b, self.m * self.rank) * std_r
                )
            self.btt_l = nn.Parameter(
                torch.randn(self.m, self.rank * self.n, self.a) * std_l
            )
        else:
            raise ValueError("init_mode must be one of: default, mup")

        # Metadata used by optimizer for the canonical new-layout BTT tensors.
        self.btt_r.btt_layout = "new"
        self.btt_r.btt_rank = self.rank
        self.btt_r.btt_b = self.b
        self.btt_r.btt_n = self.n
        self.btt_r.btt_m = self.m
        self.btt_r.btt_a = self.a

        if self.use_gate_proj:
            self.btt_g.btt_layout = "new"
            self.btt_g.btt_rank = self.rank
            self.btt_g.btt_b = self.b
            self.btt_g.btt_n = self.n
            self.btt_g.btt_m = self.m
            self.btt_g.btt_a = self.a

        self.btt_l.btt_layout = "new"
        self.btt_l.btt_rank = self.rank
        self.btt_l.btt_b = self.b
        self.btt_l.btt_n = self.n
        self.btt_l.btt_m = self.m
        self.btt_l.btt_a = self.a
        self.register_parameter("btt_s", None)

        if bias == False:
            self.register_parameter("bias", None)
        else:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return (
            f"mode: {self.decomp_mode}, init: {self.init_mode}, "
            f"blocks: ({self.m}x{self.n}), "
            f"block_size: ({self.a}x{self.b}), "
            f"rank: {self.rank}, "
            f"btt_r: {self.btt_r.shape}, "
            f"btt_g: {self.btt_g.shape if self.use_gate_proj else False}, "
            f"btt_l: {self.btt_l.shape}, "
            f"lr_act: {self.lr_act}, "
            f"lr_act_type: {self.lr_act_type}, "
            f"use_gate_proj: {self.use_gate_proj}, "
            f"bias: {self.bias.shape if self.bias is not None else False}"
        )

    @torch.no_grad()
    def init_from_linear_weight(
        self,
        weight,
        bias=None,
        s_merged_to=None,
        train_position="small",
    ):
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Linear weight shape must be {(self.out_features, self.in_features)}, "
                f"got {tuple(weight.shape)}"
            )
        if weight.shape != (self.m * self.a, self.n * self.b):
            raise ValueError(
                f"Linear weight shape {tuple(weight.shape)} not compatible with "
                f"BTT blocks (m={self.m}, n={self.n}, a={self.a}, b={self.b})"
            )

        param_dtype = weight.dtype
        # Dense weight (m*a, n*b) -> block matrix batch (m*n, a, b).
        blocks = weight.reshape(self.m, self.a, self.n, self.b)
        blocks = blocks.permute(0, 2, 1, 3).reshape(self.m * self.n, self.a, self.b)
        svd_dtype = (
            torch.float32
            if param_dtype in (torch.float16, torch.bfloat16)
            else param_dtype
        )
        U, S, Vh = torch.linalg.svd(blocks.to(dtype=svd_dtype), full_matrices=False)

        max_svd_rank = min(self.a, self.b)
        use_rank = min(self.rank, max_svd_rank)

        core_l = torch.zeros(
            self.m * self.n,
            self.a,
            self.rank,
            device=weight.device,
            dtype=param_dtype,
        )
        core_r = torch.zeros(
            self.m * self.n,
            self.rank,
            self.b,
            device=weight.device,
            dtype=param_dtype,
        )

        merge_target = resolve_blocktt_s_merged_to(
            train_position=train_position,
            s_merged_to=s_merged_to,
            left_size=self.btt_l.numel(),
            right_size=self.btt_r.numel(),
        )
        u_used = U[:, :, :use_rank].to(dtype=param_dtype)
        vh_used = Vh[:, :use_rank, :].to(dtype=param_dtype)
        s_used = torch.clamp(S[:, :use_rank], min=0).to(dtype=param_dtype)

        if merge_target == "keep":
            core_l[:, :, :use_rank] = u_used
            core_r[:, :use_rank, :] = vh_used
            s_keep = torch.zeros(
                self.m * self.n,
                self.rank,
                device=weight.device,
                dtype=param_dtype,
            )
            s_keep[:, :use_rank] = s_used
            self.btt_s = nn.Parameter(
                s_keep.reshape(self.m, self.n, self.rank), requires_grad=False
            )
        elif merge_target == "split":
            sqrt_s = torch.sqrt(s_used)
            core_l[:, :, :use_rank] = u_used * sqrt_s.unsqueeze(1)
            core_r[:, :use_rank, :] = sqrt_s.unsqueeze(-1) * vh_used
        elif merge_target == "output":
            core_l[:, :, :use_rank] = u_used * s_used.unsqueeze(1)
            core_r[:, :use_rank, :] = vh_used
        else:
            core_l[:, :, :use_rank] = u_used
            core_r[:, :use_rank, :] = s_used.unsqueeze(-1) * vh_used
        if merge_target != "keep":
            self.btt_s = None

        core_l = core_l.reshape(self.m, self.n, self.a, self.rank)
        core_r = core_r.reshape(self.m, self.n, self.rank, self.b)
        packed_l = core_l.permute(0, 1, 3, 2).reshape(
            self.m, self.rank * self.n, self.a
        )
        packed_r = core_r.permute(1, 3, 0, 2).reshape(
            self.n, self.b, self.m * self.rank
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

    @torch.no_grad()
    def materialize_dense_weight(self):
        if self.lr_act:
            raise ValueError("Dense materialization only supports lr_act=False")
        # Canonical layout:
        # btt_r: (n, b, m*r) -> (m, n, r, b)
        # btt_l: (m, n*r, a) -> (m, n, r, a)
        # W[m, a, n, b] = sum_r btt_l[m, n, r, a] * btt_r[m, n, r, b]
        r = self.btt_r.reshape(self.n, self.b, self.m, self.rank).permute(2, 0, 3, 1)
        l = self.btt_l.reshape(self.m, self.n, self.rank, self.a)
        if self.btt_s is not None:
            l = l * self.btt_s.unsqueeze(-1)
        w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)
        return w_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"BTTLayer expected last dim {self.in_features}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x = x.reshape(-1, self.n, self.b)  # (B, n, b)
        batch_n = x.shape[0]
        x_t = x.transpose(0, 1).contiguous()

        # Step 1: (n, B, b) @ (n, b, m*r) -> (n, B, m*r)
        inner_up = torch.bmm(x_t, self.btt_r)
        inner_up = inner_up.reshape(self.n, batch_n, self.m, self.rank)
        inner_up = inner_up.permute(2, 1, 0, 3).contiguous()  # (m, B, n, r)

        if self.use_gate_proj:
            inner_gate = torch.bmm(x_t, self.btt_g)
            inner_gate = inner_gate.reshape(self.n, batch_n, self.m, self.rank)
            inner_gate = inner_gate.permute(2, 1, 0, 3).contiguous()
            inner = torch.nn.functional.silu(inner_gate) * inner_up
        else:
            inner = inner_up
            if hasattr(self, "act_fn"):
                inner = self.act_fn(inner)

        # Step 2: (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        btt_l = self.btt_l
        if self.btt_s is not None:
            btt_l = (
                self.btt_l.reshape(self.m, self.n, self.rank, self.a)
                * self.btt_s.unsqueeze(-1)
            ).reshape(self.m, self.rank * self.n, self.a)
        out = torch.bmm(
            inner.reshape(self.m, batch_n, self.rank * self.n),
            btt_l,
        )
        out = out.permute(1, 0, 2).contiguous().reshape(
            *orig_shape[:-1], self.out_features
        )

        if self.bias is not None:
            out += self.bias

        return out
