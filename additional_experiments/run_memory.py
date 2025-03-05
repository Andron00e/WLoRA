import gc
import os
import sys

import torch
from transformers import HfArgumentParser, Trainer, get_scheduler

sys.path.append(os.getcwd())
from glue_experiment.utils_glue import glue_preprocess
from src import config
from src.utils import AdapterLayer


def main():
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    parser = HfArgumentParser(
        (config.ModelArguments, config.DataTrainingArguments, config.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.learning_rate = float(training_args.learning_rate)
    ################# Model, Tokenizer and Dataset Downloading #################
    if data_args.dataset_name == "glue":
        (train_dataset, eval_dataset, _, _, data_collator, _, model, tokenizer) = (
            glue_preprocess(data_args, training_args, model_args)
        )
    else:
        raise NotImplementedError("[TODO] add SuperGLUE")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    ############################ Add WLoRA Adapters ############################
    num_peft_adapters = 0
    for _, param in model.named_parameters():
        param.requires_grad = False
    if model_args.model_name_or_path == "microsoft/deberta-v3-base":
        for i in range(12):
            if num_peft_adapters < training_args.k:
                model.deberta.encoder.layer[i].attention.self.query_proj = AdapterLayer(
                    model.deberta.encoder.layer[i].attention.self.query_proj,
                    r=training_args.lora_r,
                )
                num_peft_adapters += 1
            if num_peft_adapters < training_args.k:
                model.deberta.encoder.layer[i].attention.self.key_proj = AdapterLayer(
                    model.deberta.encoder.layer[i].attention.self.key_proj,
                    r=training_args.lora_r,
                )
                num_peft_adapters += 1
            if num_peft_adapters < training_args.k:
                model.deberta.encoder.layer[i].attention.self.value_proj = AdapterLayer(
                    model.deberta.encoder.layer[i].attention.self.value_proj,
                    r=training_args.lora_r,
                )
                num_peft_adapters += 1
    else:
        raise NotImplementedError("[TODO] add LLama")
    ######################### Optimizer and Scheduler ##########################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )
    ############################# Training #####################################
    print("$" * 30, f" k={training_args.k} ", "$" * 30)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_evaluate else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=[optimizer, scheduler],
    )

    if training_args.do_train:
        _ = trainer.train()
        train_memory_gb = torch.cuda.max_memory_allocated() / 2**30

    f_name = f"./additional_experiments/memory_data/mem_"
    f_name += f"{training_args.lora_r}_{training_args.per_device_train_batch_size}"
    with open(f"{f_name}.txt", "a") as f:
        f.write(f"k={training_args.k}: {train_memory_gb}\n")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
