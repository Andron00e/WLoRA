import gc
import json
import os
import sys

import torch
import wandb
from transformers import HfArgumentParser, Seq2SeqTrainer, get_scheduler
from utils_nlg import nlg_prepocess

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
        data_collator,
        compute_metrics,
        model,
        tokenizer,
    ) = nlg_prepocess(data_args, training_args, model_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    ############################### PEFT Adapters ##############################
    all_params_before_peft, _ = utils.print_trainable_parameters(model, verbose=False)
    training_args.model_name = model_args.model_name_or_path  # for wandb
    peft_args = utils.get_peft_arguments(training_args)
    if peft_args is not None:
        model = peft.get_peft_model(model, peft_args)
    num_peft_adapters = utils.count_atapters(model, training_args.ft_strategy)
    if training_args.use_rand and training_args.ft_strategy == "WeightLoRA":
        training_args.ft_strategy = "RandLoRA"
        utils.apply_rand_weight_lora(model, num_peft_adapters, training_args.k)
    training_args.label_names = ["labels"]  # peft and compute_metrics() problem
    ft_strategy = training_args.ft_strategy
    if ft_strategy == "WeightLoRA":
        if training_args.use_fat:
            ft_strategy = "FatLoRA"
        if training_args.use_rand:
            ft_strategy = "RandLoRA"
    training_args.ft_strategy = ft_strategy
    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    if "tuned" in [training_args.learning_rate]:  # [TODO] add more tuned params
        f_name = "./nlg_experiment/tuned_params.json"
        with open(f_name) as f:
            tuned_params = json.load(f)

        lr = tuned_params[data_args.dataset_name][training_args.ft_strategy]["lr"]
        training_args.learning_rate = lr
    else:
        training_args.learning_rate = float(training_args.learning_rate)
    if training_args.ft_strategy == "FatLoRA":
        prefix = "weight_lora"
        weight_params = [
            p for name, p in model.named_parameters() if f"{prefix}_w" in name
        ]
        loraAB_params = [
            p
            for name, p in model.named_parameters()
            if f"{prefix}_A" in name or f"{prefix}_B" in name
        ]
        other_params = [p for name, p in model.named_parameters() if prefix not in name]
        optimizer = optimizers.FatAdamW(
            [
                {
                    "params": weight_params,
                    "proj": optimizers.proj_0,
                    "lr": training_args.learning_rate_w,
                    "name": "weight_params",
                },
                {"params": loraAB_params, "name": "loraAB"},
                {"params": other_params, "name": "other_params"},
            ],
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_adapters=num_peft_adapters,
            fat_step=training_args.fat_step,
            max_fat_steps=training_args.max_fat_steps,
            lora_extention=training_args.lora_extention,
        )
    elif training_args.ft_strategy == "WeightLoRA":
        weight_params = [
            p for name, p in model.named_parameters() if "weight_lora_w" in name
        ]
        others = [
            p for name, p in model.named_parameters() if "weight_lora_w" not in name
        ]
        optimizer = optimizers.WeightAdamW(
            [
                {"params": others, "name": "other_params"},
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
    run_name += f" lr={training_args.learning_rate} {data_args.dataset_name}"
    training_args.run_name = run_name
    training_args.output_dir = f"./nlg_experiment/{training_args.output_dir}/{run_name}"
    os.environ["WANDB_TAGS"] = f"NEW {data_args.dataset_name}"
    if optimizer is not None:
        training_args.optimizer = optimizer.__class__.__name__
    else:
        training_args.optimizer = training_args.optim
    training_args.benchmark_name = data_args.dataset_name
    training_args.tsk_name = data_args.task_name
    ############################# Training #####################################
    print("$" * 30, f" {run_name} ", "$" * 30)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if training_args.predict_with_generate else None
        ),
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
                            continue
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
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.val_max_target_length
        )
        num_beams = (
            data_args.num_beams
            if data_args.num_beams is not None
            else training_args.generation_num_beams
        )
        eval_metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_val_samples = (
            data_args.max_val_samples
            if data_args.max_val_samples is not None
            else len(eval_dataset)
        )
        eval_metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        eval_metrics["eval_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if "eval_runtime" in eval_metrics.keys():
            eval_metrics["eval_runtime"] /= 60
        if "wandb" in training_args.report_to:
            wandb.config.update(eval_metrics, allow_val_change=True)
    ############################################################################

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
