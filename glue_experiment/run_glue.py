import gc
import json
import os
import sys

import numpy as np
import torch
import wandb
from transformers import HfArgumentParser, Trainer, get_scheduler
from utils_glue import glue_preprocess

import peft

sys.path.append(os.getcwd())
import warnings

from src import config, optimizers, utils

warnings.filterwarnings("ignore")


def main():
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    parser = HfArgumentParser(
        (config.ModelArguments, config.DataTrainingArguments, config.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    utils.set_seed(training_args.seed)
    ################# Model, Tokenizer and Dataset Downloading #################
    (
        train_dataset,
        eval_dataset,
        test_dataset,
        datasets,
        data_collator,
        compute_metrics,
        model,
        tokenizer,
    ) = glue_preprocess(data_args, training_args, model_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    ############################### PEFT Adapters ##############################
    all_params_before_peft, _ = utils.print_trainable_parameters(model, verbose=False)
    training_args.model_name = model_args.model_name_or_path  # for wandb
    peft_args = utils.get_peft_arguments(training_args)
    if peft_args is not None:
        model = peft.get_peft_model(model, peft_args)
    for name, param in model.named_parameters():
        if (
            "attention.self" not in name
            and "output.dense" not in name
            and "intermediate.dence" not in name
        ):
            param.requires_grad = False
    num_peft_adapters = utils.count_atapters(model, training_args.ft_strategy)
    if training_args.ft_strategy == "WeightLoRA":
        if training_args.use_rand:
            training_args.ft_strategy = "RandLoRA"
            utils.apply_rand_weight_lora(model, num_peft_adapters, training_args.k)
        if training_args.use_fat:
            training_args.ft_strategy = "FatLoRA"
    training_args.label_names = ["labels"]  # peft and compute_metrics() problem
    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    if "tuned" in [training_args.learning_rate]:  # [TODO] add more tuned params
        f_name = "./glue_experiment/tuned_params.json"
        with open(f_name) as f:
            tuned_params = json.load(f)
        lr = tuned_params[data_args.task_name][training_args.ft_strategy]["lr"]
        training_args.learning_rate = lr
    else:
        training_args.learning_rate = float(training_args.learning_rate)
    if training_args.ft_strategy == "FatLoRA":
        weight_params, loraAB_params, other_params = [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "weight_lora_w" in name:
                weight_params.append(param)
            elif "weight_lora_A" in name or "weight_lora_B" in name:
                loraAB_params.append(param)
        optimizer = optimizers.FatAdamW(
            [
                {"params": loraAB_params, "name": "loraAB"},
                {"params": other_params, "name": "other_params"},
                {
                    "params": weight_params,
                    "proj": optimizers.proj_0,
                    "lr": training_args.learning_rate_w,
                    "name": "weight_params",
                },
            ],
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_adapters=len(weight_params),
            fat_step=training_args.fat_step,
            max_fat_steps=training_args.max_fat_steps,
            lora_extention=training_args.lora_extention,
        )
    elif training_args.ft_strategy == "WeightLoRA":
        weight_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "weight_lora_w" in name:
                weight_params.append(param)
            elif "weight_lora_A" in name or "weight_lora_B" in name:
                other_params.append(param)
        optimizer = optimizers.WeightAdamW(
            [
                {"params": other_params, "name": "other_params"},
                {
                    "params": weight_params,
                    "k": training_args.k,
                    "proj": optimizers.proj_0,
                    "lr": training_args.learning_rate_w,
                    "name": "weight_params",
                },
            ],
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            fat_step=training_args.fat_step,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
    ############################### Wandb Saves ################################
    training_args.all_params, training_args.trainable_params = (
        utils.print_trainable_parameters(model)
    )
    training_args.num_peft_adapters = num_peft_adapters
    training_args.peft_params = training_args.all_params - all_params_before_peft
    training_args.train_proportion = (
        training_args.trainable_params / training_args.all_params * 100
    )
    training_args.peft_proportion = (
        training_args.peft_params / training_args.all_params * 100
    )
    os.environ["WANDB_PROJECT"] = "SBER_LORA"
    if training_args.ft_strategy in ["WeightLoRA", "RandLoRA"]:
        run_name = f"[{training_args.ft_strategy} k={training_args.k} r={training_args.lora_r}]"
    else:
        run_name = f"[{training_args.ft_strategy} r={training_args.lora_r}]"
    run_name += f" {data_args.task_name}, lr={training_args.learning_rate}"
    training_args.run_name = run_name
    training_args.output_dir = (
        f"./glue_experiment/{training_args.output_dir}/{run_name}"
    )
    os.environ["WANDB_TAGS"] = f"GLUE {data_args.task_name} NEW"
    if optimizer is not None:
        training_args.optimizer = optimizer.__class__.__name__
    else:
        training_args.optimizer = training_args.optim
    training_args.benchmark_name = data_args.dataset_name
    training_args.tsk_name = data_args.task_name
    ############################# Training #####################################
    print("$" * 30, f" {run_name} ", "$" * 30)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, scheduler],
    )

    if training_args.do_train:
        train_result = trainer.train()
        train_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        train_metrics["train_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        train_metrics["train_runtime"] /= 60
        if training_args.ft_strategy in ["WeightLoRA", "RandLoRA", "FatLoRA"]:
            i = 0
            for name, param in model.named_parameters():
                if "weight_lora_w" in name:
                    if param.sum().item() > 0 and param.requires_grad:
                        i += 1
                        if training_args.model_name == "microsoft/deberta-v3-base":
                            tmp = name.split(".")
                            if "attention.self" in name:
                                layer_name = f"attn_{tmp[8].split('_')[0]}"
                            elif "attention" in name:
                                layer_name = f"attn_{tmp[7]}"
                            else:
                                layer_name = tmp[6]
                            load_name = f"{layer_name}#{tmp[5]}"
                        else:
                            load_name = name
                        train_metrics[f"active_adapters_{i}"] = load_name

        trainer.save_model()

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        if "wandb" in training_args.report_to:
            wandb.config.update(train_metrics, allow_val_change=True)
    ################################ Evaluation ################################
    if training_args.do_eval:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_val_samples = (
                data_args.max_val_samples
                if data_args.max_val_samples is not None
                else len(eval_dataset)
            )
            eval_metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
            trainer.log_metrics("Eval_%s" % task, eval_metrics)
            trainer.save_metrics("Eval_%s" % task, eval_metrics)

        if "eval_runtime" in eval_metrics.keys():
            eval_metrics["eval_runtime"] /= 60
        if "wandb" in training_args.report_to:
            wandb.config.update(eval_metrics, allow_val_change=True)
    ################################# Testing ##################################
    if training_args.do_predict:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["validation_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            max_samples = (
                data_args.max_val_samples
                if data_args.max_val_samples is not None
                else len(test_dataset)
            )
            metrics["test_samples"] = min(max_samples, len(test_dataset))
            trainer.log_metrics("Test_%s" % task, metrics)
            trainer.save_metrics("Test_%s" % task, metrics)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
