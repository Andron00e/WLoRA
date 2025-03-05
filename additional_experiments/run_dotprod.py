import gc
import os
import sys
from unittest import result

import torch
from transformers import HfArgumentParser, Trainer

sys.path.append(os.getcwd())
import warnings

from glue_experiment.utils_glue import glue_preprocess
from src import config
from src.utils import AdapterLayer, IdOptimizer

warnings.filterwarnings("ignore")


def main():
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    parser = HfArgumentParser(
        (config.ModelArguments, config.DataTrainingArguments, config.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.learning_rate = float(training_args.learning_rate)
    ################# Model, Tokenizer and Dataset Downloading #################
    train_dataset, _, _, _, data_collator, _, model, tokenizer = glue_preprocess(
        data_args, training_args, model_args
    )
    _, _, _, _, _, _, model_0, _ = glue_preprocess(data_args, training_args, model_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model_0.config.pad_token_id = model_0.config.eos_token_id
    ############################ Add WLoRA Adapters ############################
    for _, param in model.named_parameters():
        param.requires_grad = False
    for _, param in model_0.named_parameters():
        param.requires_grad = False

    name_i = training_args.k % 3
    adapter_i = training_args.k // 3
    if model_args.model_name_or_path == "microsoft/deberta-v3-base":
        if name_i == 0:
            attn_name = "query"
            model.deberta.encoder.layer[adapter_i].attention.self.query_proj = (
                AdapterLayer(
                    model.deberta.encoder.layer[adapter_i].attention.self.query_proj,
                    r=training_args.lora_r,
                )
            )
            model_0.deberta.encoder.layer[
                adapter_i
            ].attention.self.query_proj.weight.requires_grad = True
        elif name_i == 1:
            attn_name = "value"
            model.deberta.encoder.layer[adapter_i].attention.self.value_proj = (
                AdapterLayer(
                    model.deberta.encoder.layer[adapter_i].attention.self.value_proj,
                    r=training_args.lora_r,
                )
            )
            model_0.deberta.encoder.layer[
                adapter_i
            ].attention.self.value_proj.weight.requires_grad = True
        else:
            attn_name = "key"
            model.deberta.encoder.layer[adapter_i].attention.self.key_proj = (
                AdapterLayer(
                    model.deberta.encoder.layer[adapter_i].attention.self.key_proj,
                    r=training_args.lora_r,
                )
            )
            model_0.deberta.encoder.layer[
                adapter_i
            ].attention.self.key_proj.weight.requires_grad = True
    else:
        raise NotImplementedError("[TODO] add LLama")
    ######################### Optimizer and Scheduler ##########################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    optimizer_0 = IdOptimizer(
        model_0.parameters(),
    )
    ############################# Training #####################################
    run_name = f"k={training_args.k} [{attn_name}#{adapter_i}]"
    print("$" * 30, run_name, "$" * 30)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, None],
    )
    trainer_0 = Trainer(
        model=model_0,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer_0, None],
    )
    trainer.train()
    trainer_0.train()
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_A" in name:
            lora_A = param.data
        if param.requires_grad and "lora_B" in name:
            lora_B = param.data
    BA = lora_B @ lora_A
    g_0 = optimizer_0.grad_0
    # result = torch.trace(g_0.T @ BA) / (torch.linalg.norm(g_0) * torch.linalg.norm(BA))
    result = torch.trace(g_0.T @ BA)

    f_name = (
        f"./additional_experiments/dotprod_data/deberta_{data_args.task_name}_no_norm"
    )
    with open(f"{f_name}.txt", "a") as f:
        f.write(f"{run_name}: {float(result.data)}\n")

    del trainer, trainer_0, model, model_0
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
