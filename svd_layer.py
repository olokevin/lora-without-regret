import torch
import torch.nn as nn


VALID_S_MERGED_TO = {"frozen", "trainable", "output", "input", "split"}


def resolve_svd_s_merged_to(train_position, s_merged_to=None):
    if train_position not in {"output", "input", "both"}:
        raise ValueError("train_position must be one of: output, input, both")

    if s_merged_to is None:
        if train_position == "both":
            return "split"
        s_merged_to = "frozen"

    if s_merged_to not in VALID_S_MERGED_TO:
        raise ValueError(
            "s_merged_to must be one of: frozen, trainable, output, input, split"
        )

    if s_merged_to in {"output", "input", "split"}:
        return s_merged_to

    if s_merged_to == "trainable":
        if train_position == "both":
            return "split"
        return train_position

    # frozen
    if train_position == "both":
        return "split"
    return "input" if train_position == "output" else "output"


class SVDLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(in_features, out_features)

        self.svd_a = nn.Parameter(torch.empty(out_features, self.rank))
        self.svd_b = nn.Parameter(torch.empty(self.rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.svd_a, a=5**0.5)
        nn.init.kaiming_uniform_(self.svd_b, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.out_features**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def init_from_linear_weight(
        self,
        weight,
        bias=None,
        s_merged_to=None,
        train_position="output",
    ):
        compute_dtype = (
            torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        )
        u, s, vh = torch.linalg.svd(weight.to(dtype=compute_dtype), full_matrices=False)
        s = torch.clamp(s, min=0)

        merge_target = resolve_svd_s_merged_to(
            train_position=train_position,
            s_merged_to=s_merged_to,
        )
        if merge_target == "split":
            s_sqrt = torch.sqrt(s)
            a = u * s_sqrt.unsqueeze(0)
            b = s_sqrt.unsqueeze(1) * vh
        elif merge_target == "output":
            a = u * s.unsqueeze(0)
            b = vh
        else:
            a = u
            b = s.unsqueeze(1) * vh

        self.svd_a.copy_(a.to(device=weight.device, dtype=weight.dtype))
        self.svd_b.copy_(b.to(device=weight.device, dtype=weight.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=weight.device, dtype=weight.dtype))

    def materialize_dense_weight(self):
        return self.svd_a @ self.svd_b

    def forward(self, x):
        return nn.functional.linear(x, self.materialize_dense_weight(), self.bias)


def get_svd_target_module_names(svd_type):
    if svd_type == "all":
        return (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    if svd_type == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    if svd_type == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    raise ValueError("svd_type must be one of: all, mlp, attn")


@torch.no_grad()
def convert_linear_to_svd(
    model,
    skip_names=("lm_head",),
    include_names=None,
    s_merged_to=None,
    train_position="output",
):
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

    print(f"Converting {len(modules_to_replace)} Linear layers to SVD factors")

    for full_name, linear in modules_to_replace:
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]

        svd_layer = SVDLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        svd_layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
            s_merged_to=s_merged_to,
            train_position=train_position,
        )
        setattr(parent, child_name, svd_layer)

    print("Finished Linear->SVD conversion")
    return [name for name, _ in modules_to_replace]


def configure_svd_trainability(model, train_position="output", train_bias=True):
    if train_position not in {"output", "input"}:
        raise ValueError("train_position must be one of: output, input")

    for p in model.parameters():
        p.requires_grad = False

    num_svd_layers = 0
    tuned_output_cores = 0
    tuned_input_cores = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, SVDLayer):
            continue
        num_svd_layers += 1

        if train_position == "output":
            module.svd_a.requires_grad = True
            module.svd_b.requires_grad = False
            tuned_output_cores += 1
        else:
            module.svd_a.requires_grad = False
            module.svd_b.requires_grad = True
            tuned_input_cores += 1

        if module.bias is not None:
            module.bias.requires_grad = train_bias
            if train_bias:
                tuned_biases += 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_svd_layers": num_svd_layers,
        "tuned_output_cores": tuned_output_cores,
        "tuned_input_cores": tuned_input_cores,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }
