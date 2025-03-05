from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys_glue = {
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


################################ Data Arguments ################################
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default="glue", metadata={"help": "Name pf the dataset for fine-tuning"}
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys_glue.keys())
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing"},
    )
    doc_stride: Optional[int] = field(
        default=32,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    version_2_with_negative: bool = field(
        default=False,
        metadata={"help": "If true, some of the examples do not have an answer."},
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    num_beams: Optional[int] = field(
        default=True,
        metadata={"help": "[TODO]"},
    )

    def __post_init__(self):
        if self.dataset_name == "glue":
            if self.task_name is not None:
                self.task_name = self.task_name.lower()
                if self.task_name not in task_to_keys_glue.keys():
                    raise ValueError(
                        "Unknown task, you should pick one in "
                        + ",".join(task_to_keys_glue.keys())
                    )
            elif self.train_file is None or self.validation_file is None:
                raise ValueError(
                    "Need either a GLUE task or a training/validation file."
                )
            else:
                train_extension = self.train_file.split(".")[-1]
                assert train_extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
                validation_extension = self.validation_file.split(".")[-1]
                assert (
                    validation_extension == train_extension
                ), "`validation_file` should have the same extension (csv or json) as `train_file`."
        elif self.dataset_name in ["squad", "squad_v2"]:
            if (
                self.dataset_name is None
                and self.train_file is None
                and self.validation_file is None
                and self.test_file is None
            ):
                raise ValueError(
                    "Need either a dataset name or a training/validation file/test_file."
                )
            else:
                if self.train_file is not None:
                    extension = self.train_file.split(".")[-1]
                    assert extension in [
                        "csv",
                        "json",
                    ], "`train_file` should be a csv or a json file."
                if self.validation_file is not None:
                    extension = self.validation_file.split(".")[-1]
                    assert extension in [
                        "csv",
                        "json",
                    ], "`validation_file` should be a csv or a json file."
                if self.test_file is not None:
                    extension = self.test_file.split(".")[-1]
                    assert extension in [
                        "csv",
                        "json",
                    ], "`test_file` should be a csv or a json file."
        elif self.dataset_name in ["cnn_dailymail", "xsum"]:
            if (
                self.dataset_name is None
                and self.train_file is None
                and self.validation_file is None
            ):
                raise ValueError(
                    "Need either a dataset name or a training/validation file."
                )
            else:
                if self.train_file is not None:
                    extension = self.train_file.split(".")[-1]
                    assert extension in [
                        "csv",
                        "json",
                    ], "`train_file` should be a csv or a json file."
                if self.validation_file is not None:
                    extension = self.validation_file.split(".")[-1]
                    assert extension in [
                        "csv",
                        "json",
                    ], "`validation_file` should be a csv or a json file."
            if self.val_max_target_length is None:
                self.val_max_target_length = self.max_target_length

        else:
            raise ValueError(f"Wrong dataset name: {self.dataset_name}!")


############################### Model Arguments ################################
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models. Possible choises are: microsoft/deberta-v3-base, meta-llama/Meta-Llama-3.1-8B, mistralai/Mistral-7B-Instruct-v0.3, microsoft/deberta-v3-base"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,  # [TODO]
        metadata={"help": "Token Masking Probability"},
    )


############################## Training Arguments ##############################
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(
        default="./train_outputs", metadata={"help": "Folder to save output files"}
    )
    do_train: Optional[bool] = field(
        default=True, metadata={"help": "Do training or not"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "Default text key in train dataset"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=32, metadata={"help": "Batch size for training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=32, metadata={"help": "Batch size for evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=6,
        metadata={
            "help": "Number of accumulation steps to make a gradient step, i.e. before optimizer.step()"
        },
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={
            "help": "Scheduler for optimizer. We pass it in get_scheduler, possible options are"
        },
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": "Weight decay"}
    )
    eval_strategy: Optional[str] = field(
        default="steps", metadata={"help": "How to make eval"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "How often print train loss"}
    )
    seed: Optional[int] = field(
        default=18, metadata={"help": "Seed for all experiment"}
    )
    learning_rate: Optional[str] = field(
        default="8e-4",
        metadata={
            "help": "Learning rate. If `tuned`, then uses tuded hyperparametars, if float number it would be cast to float"
        },
    )
    report_to: Optional[str] = field(
        default="wandb", metadata={"help": "Where to place results. none or wandb"}
    )
    ft_strategy: Optional[str] = field(
        default="LoRA", metadata={"help": "What PEFT strategy to use"}
    )
    run_name: Optional[str] = field(
        default="zalupa", metadata={"help": "Wandb run name"}
    )
    lora_r: Optional[int] = field(
        default=None, metadata={"help": "Rank for LoRA and LoRA-like PEFT adapters"}
    )
    lora_alpha: Optional[int] = field(
        default=None, metadata={"help": "Scaling of LoRA and LoRA-like PEFT adapters"}
    )
    lora_dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout of LoRA and LoRA-like PEFT adapters"}
    )
    cls_dropout: Optional[float] = field(
        default=None, metadata={"help": "cls drop out."}
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={
            "help": "Name of the optimizer to use. Possible choises available here: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py"
        },
    )
    model_name: Optional[str] = field(default=None, metadata={"help": "For wandb"})
    optimizer: Optional[str] = field(default=None, metadata={"help": "For wandb"})
    scheduler: Optional[str] = field(default=None, metadata={"help": "For wandb"})
    benchmark_name: Optional[str] = field(default=None, metadata={"help": "For wandb"})
    tsk_name: Optional[str] = field(default=None, metadata={"help": "For wandb"})
    all_params: Optional[int] = field(
        default=None, metadata={"help": "All params of the model. For wandb"}
    )
    trainable_params: Optional[int] = field(
        default=None, metadata={"help": "Trainable params of the model. For wandb"}
    )
    train_proportion: Optional[float] = field(
        default=None,
        metadata={"help": "all_params / trainable_params * 100%. For wandb"},
    )
    peft_proportion: Optional[float] = field(
        default=None,
        metadata={
            "help": "all_params / (all_params ater PEFT - all_params before PEFT) * 100%. For wandb"
        },
    )
    peft_params: Optional[float] = field(
        default=None,
        metadata={"help": "All params after peft - All params before peft. For wandb"},
    )
    num_peft_adapters: Optional[float] = field(
        default=None, metadata={"help": "Num peft adapters across layers. For wandb"}
    )
    k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number active adapters for WeightLora. Must be less or equal to num_peft_adapters."
        },
    )
    learning_rate_w: Optional[float] = field(
        default=None, metadata={"help": "Learning rate for weights in Weight LoRA"}
    )
    compression_name: Optional[str] = field(
        default=None, metadata={"help": "ICLR KAWASAKI"}
    )
    compression_rate: Optional[float] = field(
        default=None, metadata={"help": "ICLR KAWASAKI"}
    )
    K_compress: Optional[int] = field(default=None, metadata={"help": "ICLR KAWASAKI"})
    b: Optional[float] = field(default=None, metadata={"help": "ICLR KAWASAKI"})
    use_rand: Optional[bool] = field(
        default=None,
        metadata={"help": "If true, then we use random weights in WeightLoRA"},
    )
    use_fat: Optional[bool] = field(
        default=None, metadata={"help": "If true, then we use fatness in WeightLoRA"}
    )
    fat_step: Optional[int] = field(
        default=None, metadata={"help": "Fat steps for weight lora"}
    )
    max_fat_steps: Optional[int] = field(
        default=None, metadata={"help": "Max steps of fatting of lora. For weight lora"}
    )
    lora_extention: Optional[str] = field(
        default=None,
        metadata={"help": "How to extednd adapters in FatLoRA. Can be smart ot dummy"},
    )
