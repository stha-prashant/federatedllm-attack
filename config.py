from dataclasses import dataclass, field, asdict
from typing import Optional, List
from transformers import HfArgumentParser, BitsAndBytesConfig, TrainingArguments
from trl import SFTConfig
try:
    from trl import DPOConfig
except ImportError:
    DPOConfig = None
from peft import LoraConfig
import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime
from uuid import uuid4


# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    foolsgoldbenign_reference: Optional[str] = field(default=None, metadata={"help": "the reference path for FoolsgoldBenign"})
    num_rounds: Optional[int] = field(default=500, metadata={"help": "the number of rounds"})
    # num_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})

    # FedLLM-Attack
    num_data_per_client: Optional[int] = field(default=2000, metadata={"help": "the number of data per client"})
    benign_num_clients: Optional[List[int]] = field(default=list, metadata={"help": "the numberlist of clean clients"})
    malicious_num_clients: Optional[List[int]] = field(default=list, metadata={"help": "the numberlist of malicious clients"})
    benign_dataset_names: Optional[List[str]] = field(default=list, metadata={"help": "the dataset name"})
    malicious_dataset_names: Optional[List[str]] = field(default=list, metadata={"help": "the malicious dataset name"})
    mixture_num_clients: Optional[int] = field(default=0, metadata={"help": "the number of mixture clients"})
    mixture_benign_proportions: Optional[List[float]] = field(default=list, metadata={"help": "the benign data proportions for mixture clients"})


    # safelora
    safe_lora: Optional[bool] = field(default=False, metadata={"help": "whether to use SafeLoRA to secure the aggregation"})
    evasion_lambda: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Adaptive attacker strength for safelorav2dataadaptive / "
            "safelorav2dataadaptiveold (round-delta detector): "
            "W_evade = W - λ Proj_V(W - W_global). 0 disables evasion."
        },
    )

    mixture_dirichlet_alpha: Optional[float] = field(default=0.5, metadata={"help": "the alpha parameter of the Dirichlet distribution for data partitioning in non-iid setting"})

    # smoketestanchor: fraction of each dummy client's 500 samples that come from its
    # malicious source; the remainder is BeaverTailsSafe. Length 1 applies to all
    # anchors; length N applies per-anchor. Default (empty) = fully malicious.
    smoketestanchor_poison_ratios: Optional[List[float]] = field(
        default=list,
        metadata={
            "help": "Dummy-client poison ratios for smoketestanchor "
            "(malicious_frac of 500 samples; rest BeaverTailsSafe). "
            "One value applies to all anchors, or one per anchor. "
            "Empty/default means 1.0 (fully malicious)."
        },
    )

   

@dataclass
class ScriptArguments:

    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    # dataset_name: Optional[str] = field(
    #     default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"}
    # )
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})    # vicuna and alpaca use 2e-5
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=8, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=100, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2023, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter of DPO"})
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})
    existing_lora: Optional[str] = field(default=None, metadata={"help": "the post training lora path."})

    isa: Optional[bool] = field(default=False, metadata={"help": "whether to use ISA as malicious template for attack"})

    safe_lora_original: Optional[bool] = field(default=False, metadata={"help": "whether to use the original SafeLoRA to secure the aggregation"})
    safelora_cos_thrs: Optional[List[float]] = field(default=0.35, metadata={"help": "the cosine similarity threshold for SafeLoRA"})

    normalization_method: Optional[str] = field(default="none", metadata={"help": "Normalization method (auto-filled from classifier when Safe-FedLLM is enabled; supported: 'none', 'l2')."})
    prefilter_enable: Optional[bool] = field(default=False, metadata={"help": "Enable classifier prefilter/evaluation hooks (auto-True for fed_alg=safefedllm)"})
    prefilter_classifier_path: Optional[str] = field(default=None, metadata={"help": "Path to lora classifier .pt"})
    prefilter_threshold: Optional[float] = field(default=0.8, metadata={"help": "Threshold for harmful detection configured via script arguments."})
    time_decay_factor: Optional[float] = field(default=0.95, metadata={"help": "Bayes time decay factor"})
    prefilter_min_weight: Optional[float] = field(default=0.0, metadata={"help": "Minimum soft weight for clients"})
    prefilter_log_mode: Optional[str] = field(default="json", metadata={"help": "Logging mode for prefilter weights: 'json' or 'ndjson'"})
    prefilter_strategy: Optional[str] = field(default="none", metadata={"help": "Prefilter weight strategy selector: 'step-level', 'client-level', 'shadow-level', 'evidence-level', or 'none'"})
    prefilter_round: Optional[int] = field(default=20, metadata={"help": "Only apply filtering/weight adjustment for first N rounds; then freeze"})
    prefilter_id: Optional[bool] = field(default=False, metadata={"help": "Enable dataset-specific sample_id threshold filtering (malicious<2500, benign<4000)"})
    prefilter_skip_avg_weight: Optional[float] = field(default=0.2, metadata={"help": "Skip aggregation when average client weight in a round < this threshold (vector/shadow-level)"})
    audit_interval: Optional[int] = field(default=5, metadata={"help": "Number of rounds between audits for frozen benign clients (evidence-level)"})

    analytical_alpha: Optional[float] = field(default=1.0, metadata={"help": "the alpha parameter for analytical SafeLoRA"})

    throw_n: Optional[int] = field(default=4, metadata={"help": "the index`of clients to throw away in each round for analytical SafeLoRA"})

    safelora_matrix_config: Optional[str] = field(default=None, metadata={"help": "Optional JSON file overriding SafeLoRA matrix paths per model family"})
parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

# ===== Define the LoraConfig =====
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

def get_config():
    return script_args, fed_args, peft_config


def is_dpo_dataset(dataset_name: str) -> bool:
    return "dpo" in (dataset_name or "").lower()


def is_dpo_run(fed_args) -> bool:
    names = list(fed_args.benign_dataset_names or []) + list(fed_args.malicious_dataset_names or [])
    return any(is_dpo_dataset(name) for name in names)


# ===== Define the training arguments =====
def get_training_args(script_args, new_lr, use_chat_template=False):
    sft_kwargs = dict(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        # save_steps=script_args.save_steps,
        # save_total_limit=script_args.save_total_limit,
        save_strategy='no',
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
        max_length=script_args.seq_length,
    )
    if use_chat_template:
        sft_kwargs["assistant_only_loss"] = True
        sft_kwargs["completion_only_loss"] = False
    else:
        sft_kwargs["assistant_only_loss"] = False
        sft_kwargs["completion_only_loss"] = True
    return SFTConfig(**sft_kwargs)


def get_dpo_training_args(script_args, new_lr):
    if DPOConfig is not None:
        return DPOConfig(
            output_dir=script_args.output_dir,
            per_device_train_batch_size=script_args.batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            learning_rate=new_lr,
            logging_steps=script_args.logging_steps,
            num_train_epochs=script_args.num_train_epochs,
            max_steps=script_args.max_steps,
            report_to=script_args.log_with,
            save_strategy="no",
            push_to_hub=script_args.push_to_hub,
            hub_model_id=script_args.hub_model_id,
            gradient_checkpointing=script_args.gradient_checkpointing,
            lr_scheduler_type="constant",
            remove_unused_columns=False,
            beta=script_args.dpo_beta,
            max_length=script_args.seq_length,
        )
    return TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_strategy="no",
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
        remove_unused_columns=False,
    )


def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    return device_map, quantization_config, torch_dtype

# def create_experiment_name(script_args, fed_args):
#     benign_parts = []
#     for name, num in zip(fed_args.benign_dataset_names, fed_args.benign_num_clients):
#         simplified_name = name.split('/')[-1].split('-')[0]
#         benign_parts.append(f"{simplified_name}{num}")

#     malicious_parts = []
#     for name, num in zip(fed_args.malicious_dataset_names, fed_args.malicious_num_clients):
#         simplified_name = name.split('/')[-1].split('-')[0]
#         malicious_parts.append(f"{simplified_name}{num}")
    
#     filename = "_".join(benign_parts + malicious_parts)
#     return filename


def create_experiment_name(script_args, fed_args):
    benign_parts = []
    if len(fed_args.benign_dataset_names) == len(fed_args.benign_num_clients):
        for name, num in zip(fed_args.benign_dataset_names, fed_args.benign_num_clients):
            simplified_name = name.split('/')[-1].split('-')[0]
            benign_parts.append(f"{simplified_name}{num}")
    else:
        for name in fed_args.benign_dataset_names:
            simplified_name = name.split('/')[-1].split('-')[0]
            benign_parts.append(f"{simplified_name}")

    malicious_parts = []
    for name, num in zip(fed_args.malicious_dataset_names, fed_args.malicious_num_clients):
        simplified_name = name.split('/')[-1].split('-')[0]
        malicious_parts.append(f"{simplified_name}{num}")

    proportion_parts = []
    proportions = fed_args.mixture_benign_proportions
    if proportions is not None and len(proportions) > 0:
        if len(proportions) == 1:
            values = [float(proportions[0])] * int(fed_args.mixture_num_clients or 1)
        else:
            values = [float(p) for p in proportions]
        # Compress long proportion lists (e.g. 50/100-client runs) to avoid ENAMETOOLONG.
        run_len = 1
        for i in range(1, len(values) + 1):
            if i < len(values) and values[i] == values[i - 1]:
                run_len += 1
                continue
            tag = f"{values[i - 1]:.2f}"
            proportion_parts.append(f"{run_len}x{tag}" if run_len > 1 else tag)
            run_len = 1

    filename = "_".join(benign_parts + malicious_parts + proportion_parts)
    filename = script_args.model_name_or_path.split('/')[-1] + "_" + filename
    return filename

def _create_unique_output_dir(prefix):
    while True:
        suffix = f"{datetime.now():%Y%m%d%H%M%S}_{uuid4().hex[:8]}"
        output_dir = f"{prefix}_{suffix}"
        try:
            os.makedirs(output_dir, exist_ok=False)
            return output_dir
        except FileExistsError:
            continue


def save_config(script_args, fed_args):
    dataset_name_split = create_experiment_name(script_args, fed_args)
    if script_args.existing_lora is not None:
        # Prefer malicious name when postfinetuning a malicious-only client (e.g. BeaverTails unsafe).
        benign_n = sum(fed_args.benign_num_clients) if fed_args.benign_num_clients else 0
        post_ds = (
            fed_args.benign_dataset_names[0]
            if benign_n > 0 and fed_args.benign_dataset_names
            else fed_args.malicious_dataset_names[0]
        )
        post_ds = post_ds.split('/')[-1]
        prefix = f"{script_args.output_dir}/POST-{post_ds}_num{fed_args.num_data_per_client}r{fed_args.num_rounds}s{script_args.max_steps}_{script_args.existing_lora.split('/')[-1]}"
        output_dir = _create_unique_output_dir(prefix)
    else:
        prefix = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.num_data_per_client}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}"
        output_dir = _create_unique_output_dir(prefix)
    
    script_args.output_dir = output_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
        }
        json.dump(combined_dict, f, indent=4)