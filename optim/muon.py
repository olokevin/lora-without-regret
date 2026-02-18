## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
import torch
from functools import partial
import math
import warnings
from .polar_express import PolarExpress, FastApplyPolarExpress

@torch.compile
def jiacheng(G, steps):
    """
    Jiacheng optimized polynomials
    """
    assert len(G.shape) >= 2
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    if steps > len(abc_list):
        steps = len(abc_list)
    for a, b, c in abc_list[:steps]:
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315) 
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


@torch.compile
def svd_exact_polar(G, _, cutoff=None, reverse=False):
    """
    Exact polar factorization via SVD
    """
    assert len(G.shape) >= 2
    U, Sigma, Vh = torch.linalg.svd(G.to(torch.float32), full_matrices=False)
    if cutoff is None:
        return (U @ Vh).to(G.dtype)
    else:
        Sigma = ((Sigma / Sigma.max()) >= cutoff).to(G.dtype)  # zero out small singular values
        if reverse: Sigma = 2*Sigma - 1
        return (U @ torch.diag(Sigma) @ Vh).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        lr_adam: Optional learning rate for AdamW-updated parameters.
        If None, AdamW-updated parameters share `lr`.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 lr_adam=None,
                 weight_decay=0.1,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 rms_scaling=True,
                 nuclear_scaling=False,
                 polar_method="Keller",
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 split_heads=False,
                 nheads=None,
                 polar_args={},
                 polar_params=None,
                 structured_ortho_method="mup",
                ):
        """
        Arguments:
            polar_method: The name of the polar factorization method to use (e.g., "NewtonSchultz", "Keller", "Pole") where PolE = PolarExpress
        """
        structured_ortho_method = structured_ortho_method.lower()
        allowed_structured_ortho_methods = {"mup", "svd", "naive"}
        if structured_ortho_method not in allowed_structured_ortho_methods:
            raise ValueError(
                f"Unknown structured_ortho_method: {structured_ortho_method}. "
                f"Expected one of {sorted(allowed_structured_ortho_methods)}."
            )
        defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scaling=rms_scaling,
                nuclear_scaling=nuclear_scaling,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
                structured_ortho_method=structured_ortho_method,
        )
        split_adamw_lr = lr_adam is not None
        if lr_adam is None:
            lr_adam = lr
        
        # print("EMBED TOKENS AND LM_HEAD ARE NOT HANDLED CORRECTLY FOR MUON, THEY SHOULD BE WITH ADAMW.")
        muon_params, muon_params_names = [], []
        btt_r_params, btt_r_names = [], []
        btt_l_params, btt_l_names = [], []
        adamw_params, adamw_params_names = [], []
        for name, p in named_params:
            is_excluded = any(
                excluded in name
                for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]
            )
            if "btt_r" in name:
                btt_r_params.append(p)
                btt_r_names.append(name)
            elif "btt_l" in name:
                btt_l_params.append(p)
                btt_l_names.append(name)
            elif p.ndim >= 2 and not is_excluded:
                muon_params.append(p)
                muon_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)
        muon_like_params = list(muon_params)
        muon_like_params.extend(btt_r_params)
        muon_like_params.extend(btt_l_params)
        if split_adamw_lr:
            params = []
            if len(muon_like_params) > 0:
                params.append({"params": muon_like_params, "lr": lr})
            if len(adamw_params) > 0:
                params.append({"params": adamw_params, "lr": lr_adam})
        else:
            params = muon_like_params + adamw_params
        self.split_heads = split_heads
        if self.split_heads:
            assert nheads is not None, "nheads must be specified if split_heads is True"
            self.nheads = nheads
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
        for p, p_name in zip(muon_params, muon_params_names):
            if not self.split_heads:
                assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
            if p_name.endswith("attn.c_attn.weight"):
                self.state[p]["is_W_QKV"] = True
            elif p_name.endswith("attn.c_proj.weight"):
                self.state[p]["is_W_O"] = True
        for p in btt_r_params:
            self.state[p]["use_muon"] = False
            self.state[p]["is_btt_r"] = True
        for p in btt_l_params:
            self.state[p]["use_muon"] = False
            self.state[p]["is_btt_l"] = True

        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        print("[Muon] Parameters optimized by Muon:")
        for name in muon_params_names:
            print(f"  {name}")
        print("[Muon] Parameters marked as BTT-R (custom update):")
        for name in btt_r_names:
            print(f"  {name}")
        print("[Muon] Parameters marked as BTT-L (custom update):")
        for name in btt_l_names:
            print(f"  {name}")
        print("[Muon] Parameters optimized by AdamW:")
        for name in adamw_params_names:
            print(f"  {name}")

        # Instantiate the polar factorization method
        self.polar_factorizer = self._initialize_polar_factorizer(polar_method, polar_args)
        self._vmap_status = None

    def _initialize_polar_factorizer(self, polar_method, polar_args):
        """Initialize the polar factorization method based on the provided name and parameters."""
        if polar_method == "Keller":
            return zeropower_via_newtonschulz5  # Use the method directly
        elif polar_method == "Jiacheng":
            return jiacheng
        elif polar_method == "polarexpress":
            return PolarExpress 
        elif polar_method == "fast_polarexpress":
            return partial(FastApplyPolarExpress, restart_interval=3, shift_eps=1e-3)
        elif polar_method == "svd-exact":
            return partial(svd_exact_polar, cutoff=polar_args.get("svd_cutoff", None), reverse=polar_args.get("svd_reverse", False))
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def adjust_lr_for_muon(
        self,
        lr,
        rms_scaling,
        nuclear_scaling,
        param_shape,
        grad,
        grad_sign,
        btt_scale=None,
    ):
        scale = 1.0
        if rms_scaling:
            fan_out, fan_in = param_shape[:2]
            ### original 
            scale *= math.sqrt(fan_out / fan_in)
            ### Jornan
            # scale *= math.sqrt(max(1, fan_out / fan_in))
            # Match with Kimi
            # scale *= math.sqrt(max(fan_out/fan_in, fan_in/fan_out))
            ### Kimi (not use)
            # scale *= 0.2 * math.sqrt(max(fan_out,fan_in))
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        if btt_scale is not None:
            scale *= btt_scale
        return lr * scale

    def _apply_polar_per_matrix(self, mats, steps):
        # mats: (K, M, N)
        if hasattr(torch, "vmap"):
            if self._vmap_status is None:
                try:
                    _ = torch.vmap(lambda x: self.polar_factorizer(x, steps))(mats[:1])
                    self._vmap_status = "vmap"
                    print("[Muon] BTT polar: using vmap")
                except Exception:
                    self._vmap_status = "loop"
                    print("[Muon] BTT polar: vmap failed, falling back to loop")
            if self._vmap_status == "vmap":
                return torch.vmap(lambda x: self.polar_factorizer(x, steps))(mats)
        outs = []
        for i in range(mats.shape[0]):
            outs.append(self.polar_factorizer(mats[i], steps))
        return torch.stack(outs, dim=0)

    def _adamw_update(self, p, g, group):
        lr = group["lr"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        weight_decay = group["weight_decay"]
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["moment1"] = torch.zeros_like(g)
            state["moment2"] = torch.zeros_like(g)
        state["step"] += 1
        step = state["step"]
        buf1 = state["moment1"]
        buf2 = state["moment2"]
        buf1.lerp_(g, 1 - beta1)
        buf2.lerp_(g.square(), 1 - beta2)

        g = buf1 / (eps + buf2.sqrt())

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        scale = bias_correction1 / bias_correction2**0.5
        p.data.mul_(1 - lr * weight_decay)
        p.data.add_(g, alpha=-lr / scale)

    def step(self, closure=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
"""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                        
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                g = p.grad
                if g is None:
                    continue
                if (g.ndim > 2) and not (self.split_heads):
                    g = g.view(g.size(0), -1)

                assert g is not None
                
                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if self.split_heads and self.state[p].get("is_W_QKV", False):
                    # For W_QKV, we split the gradients into 3 heads and process them separately
                    # print("before", g.shape, self.nheads)
                    old_shape = g.shape
                    g = g.reshape(3 * self.nheads, g.shape[0] // (3 * self.nheads), g.shape[1])
                    # print("after", g.shape)
                elif self.split_heads and self.state[p].get("is_W_O", False) and self.split_heads:
                    # print("before", g.shape, self.nheads)
                    old_shape = g.shape
                    g = g.reshape(g.shape[0], self.nheads, g.shape[1] // self.nheads).transpose(0, 1)
                    # print("after", g.shape)
                    # For W_O, we split the gradients into 3 heads and process them separately

                # Use the selected polar factorization method. u is always in bfloat16 for efficiency.
                u = self.polar_factorizer(g, group["ns_steps"])
                
                if self.split_heads and self.state[p].get("is_W_QKV", False):
                    g = g.reshape(old_shape)
                    u = u.reshape(old_shape)
                elif self.split_heads and self.state[p].get("is_W_O", False):
                    g = g.transpose(0, 1).reshape(old_shape)
                    u = u.transpose(0, 1).reshape(old_shape)

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(
                    lr,
                    group["rms_scaling"],
                    group["nuclear_scaling"],
                    p.shape,
                    g.bfloat16(),  # convert to float16 to be compatible with u
                    u
                )
                
                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)
                
                # apply update
                p.data.add_(u, alpha=-adjusted_lr)
                
            ############################
            #         BTT Muon         #
            ############################

            btt_params = [
                p
                for p in group["params"]
                if self.state[p].get("is_btt_r", False)
                or self.state[p].get("is_btt_l", False)
            ]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in btt_params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if state.get("is_btt_r", False):
                    method = group["structured_ortho_method"]
                    if g.dim() == 4:
                        # Old layout: (m, n, r, b)
                        m, n, r, b = g.shape
                        if method == "mup":
                            mats = g.permute(1, 0, 2, 3).reshape(n, m * r, b)
                            u = (
                                self._apply_polar_per_matrix(mats, group["ns_steps"])
                                .reshape(n, m, r, b)
                                .permute(1, 0, 2, 3)
                            )
                            btt_scale = math.sqrt(m * r / b)
                        elif method == "svd":
                            mats = g.reshape(-1, r, b)
                            u = self._apply_polar_per_matrix(
                                mats, group["ns_steps"]
                            ).reshape(m, n, r, b)
                            btt_scale = math.sqrt(r / b / n)
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.transpose(1, 2).reshape(m * r, n * b),
                                group["ns_steps"],
                            ).reshape(m, r, n, b).transpose(1, 2)
                            btt_scale = math.sqrt((m * r) / (n * b))
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    elif g.dim() == 3:
                        # New layout: (n, b, m*r)
                        n, b, mr = g.shape
                        if method == "mup":
                            u = self._apply_polar_per_matrix(
                                g, group["ns_steps"]
                            )
                            btt_scale = math.sqrt(mr / b)
                        elif method == "svd":
                            rank = getattr(p, "btt_rank", None)
                            ns0 = getattr(p, "btt_ns0", None)
                            if rank is None or ns0 is None or rank * ns0 != mr:
                                mats = g.permute(0, 2, 1)
                                u = self._apply_polar_per_matrix(
                                    mats, group["ns_steps"]
                                ).permute(0, 2, 1)
                                btt_scale = math.sqrt(mr / b)
                            else:
                                mats = g.permute(0, 2, 1).reshape(n, ns0, rank, b)
                                mats = mats.reshape(n * ns0, rank, b)
                                u = self._apply_polar_per_matrix(mats, group["ns_steps"])
                                u = u.reshape(n, ns0, rank, b).reshape(n, mr, b)
                                u = u.permute(0, 2, 1)
                                btt_scale = math.sqrt(rank / b / n)
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.permute(2, 0, 1).reshape(mr, n * b),
                                group["ns_steps"],
                            ).reshape(mr, n, b).permute(1, 2, 0)
                            btt_scale = math.sqrt(mr / (n * b))
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    else:
                        self._adamw_update(p, g, group)
                        continue
                elif state.get("is_btt_l", False):
                    method = group["structured_ortho_method"]
                    if g.dim() == 4:
                        # Old layout: (m, n, a, r)
                        m, n, a, r = g.shape
                        if method == "mup":
                            mats = g.permute(0, 2, 1, 3).reshape(m, a, n * r)
                            u = (
                                self._apply_polar_per_matrix(mats, group["ns_steps"])
                                .reshape(m, a, n, r)
                                .permute(0, 2, 1, 3)
                            )
                            btt_scale = math.sqrt(a / (n * r))
                        elif method == "svd":
                            mats = g.reshape(-1, a, r)
                            u = self._apply_polar_per_matrix(
                                mats, group["ns_steps"]
                            ).reshape(m, n, a, r)
                            btt_scale = math.sqrt(a / r / n)
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.transpose(1, 2).reshape(m * a, n * r),
                                group["ns_steps"],
                            ).reshape(m, a, n, r).transpose(1, 2)
                            btt_scale = math.sqrt((m * a) / (n * r))
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    elif g.dim() == 3:
                        # New layout: (m, n*r, a)
                        m, nr, a = g.shape
                        if method == "mup":
                            u = self._apply_polar_per_matrix(
                                g, group["ns_steps"]
                            )
                            btt_scale = math.sqrt(a / nr)
                        elif method == "svd":
                            rank = getattr(p, "btt_rank", None)
                            ms1 = getattr(p, "btt_ms1", None)
                            if rank is None or ms1 is None or rank * ms1 != nr:
                                mats = g.transpose(1, 2)
                                u = self._apply_polar_per_matrix(
                                    mats, group["ns_steps"]
                                ).transpose(1, 2)
                                btt_scale = math.sqrt(a / nr)
                            else:
                                mats = g.reshape(m, ms1, rank, a).permute(0, 1, 3, 2)
                                mats = mats.reshape(m * ms1, a, rank)
                                u = self._apply_polar_per_matrix(mats, group["ns_steps"])
                                u = u.reshape(m, ms1, a, rank).permute(0, 1, 3, 2)
                                u = u.reshape(m, nr, a)
                                btt_scale = math.sqrt(a / nr)
                        elif method == "naive":
                            u = self.polar_factorizer(
                                g.transpose(1, 2).reshape(m * a, nr),
                                group["ns_steps"],
                            ).reshape(m, a, nr).transpose(1, 2)
                            btt_scale = math.sqrt((m * a) / nr)
                        else:
                            raise ValueError(f"Unknown structured_ortho_method: {method}")
                    else:
                        self._adamw_update(p, g, group)
                        continue
                else:
                    # Unknown BTT tensor layout; fall back to AdamW for safety.
                    self._adamw_update(p, g, group)
                    continue

                adjusted_lr = self.adjust_lr_for_muon(
                    lr,
                    False,
                    False,
                    (1, 1),
                    g.bfloat16(),
                    u,
                    btt_scale=btt_scale,
                )

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [
                p
                for p in group["params"]
                if (not self.state[p]["use_muon"])
                and (not self.state[p].get("is_btt_r", False))
                and (not self.state[p].get("is_btt_l", False))
            ]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                self._adamw_update(p, g, group)
                    
        return loss
