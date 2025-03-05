import json
import logging
import os
import random

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from sympy import im
from transformers import (AutoConfig, AutoModelForQuestionAnswering,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EvalPrediction,
                          PretrainedConfig, PreTrainedTokenizerFast,
                          default_data_collator)

from src import utils_qa

logger = logging.getLogger(__name__)
# Loading MMLU categories
if not os.path.exists("./categories.py"):
    os.system(
        "wget https://raw.githubusercontent.com/hendrycks/test/master/categories.py"
    )
from src.categories import categories as categories_inv
from src.categories import subcategories


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


def load_and_transform_jsonl(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    transformed_data = [{"chat": item["chat"]} for item in data]
    return transformed_data


def set_seed(seed):  # ставит сид
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device(device_no: int):  # выбирает GPU-шку и выводит название
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
        for name, _ in model.named_parameters():
            if adapter_name in name:
                num_adapters += 1

    return num_adapters


def make_mlm_dataset_form_mmlu(dataset):
    dataset_list = []
    for a in dataset:
        q = a["question"]
        q = q.replace("_", "")
        q += " " + a["choices"][a["answer"]]
        q = q.replace(".", "")
        q = q.replace("  ", " ")
        q += "."
        dataset_list.append({"text": q})

    return_dataset = datasets.Dataset.from_list(dataset_list)
    return return_dataset


def mmlu_preporcess(config):
    for subcat_name, cat_names in subcategories.items():
        subcategories[subcat_name] = (
            cat_names[0] if isinstance(cat_names, list) else cat_names
        )
    categories = {}
    for cat_name, subcats in categories_inv.items():
        for subcat in subcats:
            categories[subcat] = cat_name

    def subcat_to_cat(subcat):
        cat_name = subcategories[subcat]
        cat_name = categories[cat_name]

        return cat_name

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side=config.tokenizer_config.padding_side,
        model_max_length=config.max_length,
        use_fast=False,
    )
    # tokenizer_mistral = transformers.AutoTokenizer.from_pretrained(
    #     "mistralai/Mistral-7B-Instruct-v0.3",
    #     padding_side=config.tokenizer_config.padding_side,
    # #     model_max_length=512,
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template="llama-3"
    # )
    # print(tokenizer.chat_template) # = "<prompt_template>"
    # tokenizer.chat_template = tokenizer_mistral.chat_template
    EOS_TOKEN = tokenizer.eos_token
    mmlu_dataset = load_dataset("cais/mmlu", config.task_name)

    few_shot_datasets = {
        subject: mmlu_dataset["dev"].filter(lambda row: row["subject"] == subject)
        for subject in set(mmlu_dataset["dev"]["subject"])
    }

    def prepare_question(examples):
        prompt = f"{examples['question']}\n"
        for letter, choice in zip(("A", "B", "C", "D"), examples["choices"]):
            prompt += f"{letter}. {choice}\n"

        answer = chr(65 + examples["answer"])

        return prompt, answer

    def prepare_prompt(examples, dev_dataset=None):
        if dev_dataset:
            yield from map(prepare_question, dev_dataset)

        yield prepare_question(examples)

    def prepare_instruction_text(example):
        instructions = [
            {
                "role": "system",
                "content": f"The following are multiple choice questions (with answers) about {example['subject']}. Output 'A', 'B', 'C', or 'D'. Full answer not needed.",
            },
        ]

        if config["n_shots"] and example["subject"]:
            few_shot_dataset = few_shot_datasets[example["subject"]]
            few_shot_dataset = few_shot_dataset.select(range(config["n_shots"]))
        else:
            few_shot_dataset = None

        for prompt, ans in prepare_prompt(example, dev_dataset=few_shot_dataset):
            instructions.append({"role": "user", "content": prompt})
            instructions.append({"role": "assistant", "content": ans})

        text = tokenizer.apply_chat_template(instructions, tokenize=False)

        return {"text": text}

    def r_replace(line, old, new):
        return line[::-1].replace(old[::-1], new[::-1], 1)[::-1]

    def remove_answer(example):
        text_wa_answer = example["text"]
        text_wa_answer = text_wa_answer.rsplit("<|eot_id|>", 1)[0][:-1]

        return {"text_wa_answer": text_wa_answer}

    instructions_datasets = mmlu_dataset.map(
        prepare_instruction_text, batched=False, num_proc=2
    )
    instructions_datasets["validation"] = instructions_datasets["validation"].map(
        remove_answer, batched=False
    )
    instructions_datasets["test"] = instructions_datasets["test"].map(
        remove_answer, batched=False
    )

    instructions_datasets.set_format("torch")

    # Accessing the train, validation, and test splits
    validation_dataset = instructions_datasets["validation"]
    test_dataset = instructions_datasets["test"]
    dev_dataset = instructions_datasets["dev"]  # dataset for few shot
    auxiliary_train_dataset = instructions_datasets["auxiliary_train"]

    # Check the size of each split
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Auxiliary train dataset size: {len(auxiliary_train_dataset)}")

    return auxiliary_train_dataset, test_dataset, validation_dataset, dev_dataset


def mmlu_process_prediction(pred):
    pred = pred["generated_text"]

    pred = pred.strip().upper()

    pred = pred[0] if pred else "I"
    pred = pred if pred in {"A", "B", "C", "D"} else "I"

    return pred


def glue_preprocess(data_args, training_args, model_args):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        raise RuntimeError("Pass the data_args.task_name !!!")

    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    train_dataset = None
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    eval_dataset = None
    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets[
            "validation_matched" if data_args.task_name == "mnli" else "validation"
        ]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    test_dataset = None
    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Get the metric function
    metric = load_metric("glue", data_args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif data_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    return (
        train_dataset,
        eval_dataset,
        test_dataset,
        datasets,
        data_collator,
        compute_metrics,
        model,
        tokenizer,
    )


def squad_preprocess(data_args, training_args, model_args):
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_dropout=training_args.cls_dropout,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_token="[CLS]",
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [
            q.lstrip() for q in examples[question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [
            q.lstrip() for q in examples[question_column_name]
        ]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    eval_dataset, eval_examples = None, None
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_val_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_val_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_val_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_val_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    test_dataset, test_examples = None, None
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(
                range(data_args.max_predict_samples)
            )
        # Predict Feature Creation
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            test_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = utils_qa.postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    return (
        train_dataset,
        eval_dataset,
        eval_examples,
        test_dataset,
        test_examples,
        data_collator,
        compute_metrics,
        model,
        tokenizer,
        post_processing_function,
    )


def apply_rand_weight_lora(model, n, k):
    idxs = torch.randperm(n)[:k]
    i = 0
    for name, param in model.named_parameters():
        if "weight_lora_w" in name:
            param.requires_grad = False
            if i not in idxs:
                param.data = torch.tensor([0.0])
            i += 1
