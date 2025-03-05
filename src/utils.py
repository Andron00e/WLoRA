import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

import peft


def print_trainable_parameters(model, verbose=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    for param in model.buffers():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if verbose:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    return all_param, trainable_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(device_no: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device


def count_atapters(model, peft_type):
    if peft_type in ["LoRA", "ADALoRA", "DoRA", "rsLoRA"]:
        adapter_name = "lora_A"
    elif peft_type == "LoKR":
        adapter_name = "lokr_w1"
    elif peft_type == "LoHA":
        adapter_name = "hada_w1_a"
    elif peft_type == "VERA":
        adapter_name = "vera_lambda_b"
    elif peft_type in ["WeightLoRA", "RandLoRA"]:
        adapter_name = "weight_lora_A"
    elif peft_type == "Full":
        adapter_name = None
    else:
        raise ValueError(f"Wrong peft_type: {peft_type}")

    num_adapters = None
    if adapter_name is not None:
        num_adapters = 0
        for name, param in model.named_parameters():
            if adapter_name in name and param.requires_grad:
                num_adapters += 1

    return num_adapters


def apply_rand_weight_lora(model, n, k):
    idxs = torch.randperm(n)[:k]
    i = 0
    for name, param in model.named_parameters():
        if "weight_lora_w" in name:
            param.requires_grad = False
            if i not in idxs:
                param.data = torch.tensor([0.0])
            i += 1


def get_peft_arguments(training_args):
    if training_args.ft_strategy == "LoRA":
        peft_args = peft.LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
    elif training_args.ft_strategy == "LoKR":
        peft_args = peft.LoKrConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
    elif training_args.ft_strategy == "LoHA":
        peft_args = peft.LoHaConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
    elif training_args.ft_strategy == "VERA":
        peft_args = peft.VeraConfig(
            r=training_args.lora_r, vera_dropout=training_args.lora_dropout
        )
    elif training_args.ft_strategy == "ADALoRA":
        peft_args = peft.AdaLoraConfig(
            target_r=training_args.lora_r,
        )
    elif training_args.ft_strategy == "DoRA":
        peft_args = peft.LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            use_dora=True,
        )
    elif training_args.ft_strategy == "rsLoRA":
        peft_args = peft.LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            use_rslora=True,
        )
    elif training_args.ft_strategy == "WeightLoRA":
        peft_args = peft.WeightLoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
    elif training_args.ft_strategy == "Full":
        return None
    else:
        raise ValueError(f"Incorrect FT type {training_args.ft_strategy}!")

    if training_args.model_name in ["microsoft/deberta-v3-base"]:
        # peft_args.target_modules = ["query_proj", "key_proj", "value_proj",
        #                             "intermediate.dence", "output.dence"]
        peft_args.target_modules = "all-linear"
    elif training_args.model_name in ["facebook/bart-large"]:
        # peft_args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                             "gate_proj", "up_proj", "down_proj",
        #                             "fc1", "fc2"]
        peft_args.target_modules = "all-linear"
    else:
        raise ValueError(
            f"Pass target_modules to your model {training_args.model_name}"
        )
    return peft_args


class AdapterLayer(nn.Module):
    """Wraps a linear layer with LoRA-like adapter. Wraps an existing OPT linear layer"""

    def __init__(self, module: nn.Linear, r: int = 8, add_weight=True):
        super().__init__()
        self.module = module  # pre-trained (frozen) linear layer
        self.lora_A = nn.Linear(
            in_features=module.in_features,
            out_features=r,
            bias=False,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        self.lora_B = nn.Linear(
            in_features=r,
            out_features=module.out_features,
            bias=False,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        nn.init.zeros_(self.lora_B.weight)
        self.add_weight = add_weight
        if add_weight:
            self.w = torch.tensor(1.0, requires_grad=True)

    def forward(self, x):
        frwd_module = self.module(x)
        if self.add_weight:
            frwd_adapter = self.w * self.lora_B(self.lora_A(x))
        else:
            frwd_adapter = self.lora_B(self.lora_A(x))
        return frwd_module + frwd_adapter


class IdOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(IdOptimizer, self).__init__(params, defaults)
        self.grad_0 = None

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.grad_0 is None:
                    self.grad_0 = p.grad.data
                else:
                    self.grad_0 += p.grad.data
        return loss
