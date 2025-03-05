import math
import warnings
from typing import Callable

import numpy as np
import torch
import torch.optim as optim

############################### torch optimizers ###############################
"""
LAMB optimizer: torch_optimizer.Lamb(model.parameters(), lr=0.001)
LAMB scheduler: vanilla linear (?) 
Prodigy optimizer: prodigyopt.Prodigy(model.parameters(), lr=1.)
Prodijy scheduler: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
DAdaptAdam optimizer: dadaptation.DAdaptAdam(model.parameters(), lr=1.)
DAdaptAdam scheduler: same as for Prodijy optimizer (?)
"""
################################################################################
########### hugging face scheduler
"""
    - "linear" = get_linear_schedule_with_warmup
    - "cosine" = get_cosine_schedule_with_warmup
    - "cosine_with_restarts" = get_cosine_with_hard_restarts_schedule_with_warmup
    - "polynomial" = get_polynomial_decay_schedule_with_warmup
    - "constant" =  get_constant_schedule
    - "constant_with_warmup" = get_constant_schedule_with_warmup
    - "inverse_sqrt" = get_inverse_sqrt_schedule
    - "reduce_lr_on_plateau" = get_reduce_on_plateau_schedule
    - "cosine_with_min_lr" = get_cosine_with_min_lr_schedule_with_warmup
    - "warmup_stable_decay" = get_wsd_schedule
"""


# from https://github.com/jxbz/signSGD/blob/master/signSGD_zeros.ipynb
class signSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr)
        self.rand_zero = rand_zero
        super(signSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # take sign of gradient
                grad = torch.sign(p.grad)
                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad == 0] = (
                        torch.randint_like(grad[grad == 0], low=0, high=2) * 2 - 1
                    )
                    assert not (grad == 0).any()
                # make update
                p.data -= group["lr"] * grad
        return loss


class signAdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        rand_zero: bool = True,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        self.rand_zero = rand_zero
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                ########################## sign step ###########################
                # OLD: grad = p.grad
                grad = torch.sign(p.grad)
                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad == 0] = (
                        torch.randint_like(grad[grad == 0], low=0, high=2) * 2 - 1
                    )
                    assert not (grad == 0).any()
                ################################################################
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


class AdamW(optim.Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        return loss


class SGD(optim.Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data -= group["lr"] * p.grad.data
        return loss


class StoIHT(optim.Optimizer):
    def __init__(self, params, k, approx, proj, prob, lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, approx=approx, proj=proj, k=k, prob=prob)
        super(StoIHT, self).__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                b_t = p.data - group["lr"] * d_p
                if np.random.random() < group["prob"]:
                    Gamma_t = group["approx"](b_t, group["k"])
                    p.data = group["proj"](b_t, Gamma_t)
                else:
                    p.data = b_t
        return loss


def approx_0(x, k):
    if len(x.shape) == 1:
        idxs = torch.sort(x, descending=True).indices[:k]
        mask = torch.zeros_like(x, dtype=x.dtype)
        mask[idxs] = 1.0
        return mask
    elif len(x.shape) == 2:
        x_long = x.reshape(-1)
        idxs = torch.sort(x_long, descending=True).indices[:k]
        mask = torch.zeros_like(x_long, dtype=x.dtype)
        mask[idxs] = 1.0
        return mask.reshape(x.shape)
    else:
        raise NotImplementedError("Only x.shape() 1 and 2 available!")


def proj_0_old(x, mask):
    return x.mul(mask)


def proj_0(x: torch.Tensor, K: int):
    _, idx = torch.topk(x, k=x.shape[0] - K, largest=False)
    x[idx] = 0.0
    return x


class FatAdamW(optim.Optimizer):
    """
    Implements Adam algorithm with weight decay for Weight Lora adapter

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        num_adapters: int = 36,
        lora_extention: str = "dummy",
        fat_step: int = 10,
        max_fat_steps: int = 3,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.k = num_adapters
        self.chosen_layers = list(range(num_adapters))
        if lora_extention not in ["smart", "dummy", "restart"]:
            raise ValueError(f"Wrong lora_extention: {lora_extention}")
        self.lora_extention = lora_extention
        self.fat_step = fat_step
        self.max_fat_steps = max_fat_steps
        self.R, self.R_mom, self.R_mom_sq = None, None, None

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["name"] != "weight_params":
                ############################ Adam Step #############################
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        continue
                    if (i // 2) not in self.chosen_layers and group["name"] == "loraAB":
                        continue

                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    if (
                        group["name"] == "loraAB"
                        and state["step"] % self.fat_step == 0
                        and self.max_fat_steps >= 0
                    ):
                        A_or_B = np.argmin(p.data.shape)
                        # self.state[p] = {}
                        if A_or_B == 1:  # lora_A
                            if self.lora_extention == "dummy":
                                N = torch.rand_like(p.data, requires_grad=True)
                                p.data = torch.concat([p.data, N], dim=1)
                            elif self.lora_extention == "smart":
                                Q, self.R = torch.linalg.qr(p.data, mode="reduced")
                                N = torch.rand_like(p.data, requires_grad=True)
                                I = torch.eye(
                                    np.max(p.data.shape),
                                    requires_grad=True,
                                    device=p.data.device,
                                )
                                p.data = torch.concat([Q, (I - Q @ Q.T) @ N], dim=1)
                            elif self.lora_extention == "restart":
                                N_1 = torch.rand_like(p.data, requires_grad=True)
                                N_2 = torch.rand_like(p.data, requires_grad=True)
                                p.data = torch.concat([N_1, N_2], dim=1)
                        else:  # lora_B
                            O = torch.zeros_like(p.data, requires_grad=True)
                            if self.lora_extention == "dummy":
                                p.data = torch.concat([p.data, O], dim=0)
                            elif self.lora_extention == "smart":
                                p.data = torch.concat([self.R @ p.data, O], dim=0)
                            elif self.lora_extention == "restart":
                                p.data = torch.concat([O, O], dim=0)

                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        continue

                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
            else:
                ######################## StoIHT step for w #########################
                if self.max_fat_steps == 0:
                    self.max_fat_steps -= 1
                if self.max_fat_steps > 0:
                    w_vector = []
                    if "w_step" not in group.keys():
                        group["w_step"] = 0
                    group["w_step"] += 1
                    for i, p in enumerate(group["params"]):
                        if p.grad is None or i not in self.chosen_layers:
                            continue
                        p.add_(p.grad, alpha=-group["lr"])
                        if group["w_step"] % self.fat_step == 0:
                            w_vector.append(p.data.item())

                    if group["w_step"] % self.fat_step == 0:
                        self.k //= 2
                        self.max_fat_steps -= 1
                        new_chosen_layers = []
                        w_vector = torch.tensor(w_vector)
                        w_vector = group["proj"](w_vector, self.k)
                        j = 0
                        for i, p in enumerate(group["params"]):
                            if p.grad is None or i not in self.chosen_layers:
                                continue
                            if w_vector[j] > 0:
                                new_chosen_layers.append(i)
                            p.data = torch.tensor([w_vector[j]], device=p.device)
                            j += 1
                        self.chosen_layers = new_chosen_layers
                        print("$$$$$$$$", self.chosen_layers, "$$$$$$$$")
            ####################################################################
        return loss


class WeightAdamW(optim.Optimizer):
    """
    Implements Adam algorithm with weight decay for Weight Lora adapter

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        fat_step: int = 1,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.fat_step = fat_step

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["name"] != "weight_params":
                ############################ Adam Step #############################
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    step_size = group["lr"]
                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = (
                            step_size * math.sqrt(bias_correction2) / bias_correction1
                        )

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group["weight_decay"] > 0.0:
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
            else:
                ######################## StoIHT step for w #########################
                w_vector = []
                w_grad = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                    state["step"] += 1
                    p.add_(p.grad, alpha=-group["lr"])
                    w_grad.append(p.grad.item())
                    w_vector.append(p.data.item())
                j = 0
                w_grad = torch.tensor(w_grad)
                if (
                    state["step"] % self.fat_step == 0
                    and torch.linalg.norm(w_grad).item() > 1e-10
                ):
                    w_vector = torch.tensor(w_vector)
                    w_vector = group["proj"](w_vector, group["k"])
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        p.data = torch.tensor([w_vector[j]], device=p.device)
                        j += 1
            ####################################################################
        return loss


class WeightAdamW_old(optim.Optimizer):
    """
    Implements Adam algorithm with weight decay for Weight Lora adapter

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        calculate_w_step: int = 1,
        k: int = 100,
        approx=approx_0,
        proj=proj_0_old,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        self.k = k
        self.approx = approx
        self.proj = proj
        self.calculate_w_step = calculate_w_step
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            w_data = []
            w_grad = []
            sum = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.data.shape == torch.Size([1]):
                    # print(f"Before w step: w: {p.data.item()}, grad_w = {p.grad.item()}")
                    w_data.append(p.data.item())
                    w_grad.append(p.grad.item())
                    continue

                grad = p.grad
                if grad.norm() > 0:
                    sum += 1
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                p.addcdiv_(exp_avg, denom, value=-step_size)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

            # print(f"There are {sum} non-null gradients")
            ######################## StoIHT step for w #########################
            # w_i = w_i / (w_i + eps)
            w_data = torch.tensor(w_data)
            w_grad = torch.tensor(w_grad)
            if w_grad.norm() > 0:
                b_t = w_data - group["lr"] * w_grad
                # b_t = w_data - w_grad
                Gamma_t = self.approx(b_t, self.k)
                w_data = self.proj(b_t, Gamma_t)
                for a in w_data:
                    print(a)
                i = 0
                for p in group["params"]:
                    if p.data.shape == torch.Size([1]):
                        p.data = torch.tensor([w_data[i]], device=p.device)
                        i += 1
                        # print(f"i={i}, p.data={p.data}")
                        # print(f"After w step: w: {p.data.item()}")
                # print("$"*60)
            ####################################################################
        return loss


from math import ceil


class Rand:
    def __init__(self, compressor_params: dict = {"compression_rate": 0.1}) -> None:
        self.compression_rate = compressor_params["compression_rate"]
        self.used_coordinates = []
        self.K = 0

    def get_probs(self, x: torch.Tensor):
        return torch.ones_like(x)

    def compress(self, x: torch.Tensor):
        d = x.shape[0]
        m = ceil(self.compression_rate * d)
        x_flatten = x.reshape(-1)
        probs = self.get_probs(x_flatten)
        idxs = torch.multinomial(probs, m)
        x_flatten[~idxs] = 0
        self.used_coordinates = idxs.tolist() + self.used_coordinates
        self.used_coordinates = self.used_coordinates[: self.K * m]
        return 1.0 / self.compression_rate * x_flatten.reshape(x.shape)


class Banlast(Rand):
    def __init__(self, compressor_params: dict = {"compression_rate": 0.1, "K": 7}):
        super().__init__(compressor_params)
        self.K = compressor_params["K"]

    def get_probs(self, x: torch.Tensor):
        probs = torch.ones_like(x)
        probs[self.used_coordinates] = 0.0
        if probs.sum().item() == 0:
            probs = torch.ones_like(x)
        return probs


class KAWASAKI(Rand):
    def __init__(
        self,
        compressor_params: dict = {
            "compression_rate": 0.1,
            "K": 7,
            "b": 2.0,
            "proj": None,
        },
    ):
        super().__init__(compressor_params)
        self.K = compressor_params["K"]
        self.b = compressor_params["b"]
        self.proj = compressor_params["proj"]

    def get_probs(self, x: torch.Tensor):
        probs = torch.ones_like(x)
        num_used_coordinates = torch.Tensor(
            [self.used_coordinates.count(i) for i in range(x.shape[0])]
        ).to(x.device)
        probs /= self.b**num_used_coordinates
        if self.proj is not None:
            probs = self.proj(probs)
        return probs


class QSGD(optim.Optimizer):
    def __init__(
        self, params, lr=0.01, compression_name: str = None, compressor_params=None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(QSGD, self).__init__(params, defaults)
        if compression_name not in ["Rand", "BanLast", "KAWASAKI", None]:
            raise ValueError(f"Wrong compression name {compression_name}")
        self.compression_name = compression_name
        self.compressor_params = compressor_params

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    if self.compression_name == "Rand":
                        state["compressor"] = Rand(self.compressor_params)
                    elif self.compression_name == "BanLast":
                        state["compressor"] = Banlast(self.compressor_params)
                    elif self.compression_name == "KAWASAKI":
                        state["compressor"] = KAWASAKI(self.compressor_params)
                    else:
                        state["compressor"] = None
                if state["compressor"] is not None:
                    grad = state["compressor"].compress(p.grad.data)
                    p.data -= group["lr"] * grad
                else:
                    p.data -= group["lr"] * p.grad.data
        return loss
