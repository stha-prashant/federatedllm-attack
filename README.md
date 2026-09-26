For our implementation please look at `federated_learning/ours.py` file.

## Setup

```bash
conda env create -f environment.yml
conda activate testvllm
export HUGGINGFACE_HUB_TOKEN=...
```

## Training Unsafe Model

```bash
bash training_scripts/train_reference.sh
```

This will save final checkpoint to a path `<model_directory>/checkpoint-50`

## Compute reference unsafe direction

Find the path of final checkpoint saved of the unsafe model as  `<model_directory>/checkpoint-50` and merge the LoRA weights to obtain the full model.

```bash
python utils/merge_lora.py --base_model_path meta-llama/Llama-2-7b-chat-hf --lora_path <model_directory>/checkpoint-50
```

This will result in creation of `<model_directory>/full-50`. Use this path and run

```bash
python compute_direction.py --harmful_checkpoint_path `<model_directory>/full-50`
```

This will save a `.pkl` file to a path. Use that full path and edit `configs/safelora_matrix_paths.json` and set `delta_harmful_systemprompt` to your harmful-direction delta matrix (`.pkl`) path for the target model family (Llama2 by default).

## Run Ours

Defaults: Llama-2-7B-Chat, seed `2023`, 10 clients with **3 malicious** at poison fraction **0.5**, Dirichlet α `0.5`, malicious data `PKU-Alignment/BeaverTails`, utility data PubMedQA/medQA/emrqa/cord19.

```bash
export HUGGINGFACE_HUB_TOKEN=...
# edit configs/safelora_matrix_paths.json first
sbatch training_scripts/run_ours.sh
```

## Baselines

Change `--fed_alg` to one of : `fedavg`, `median`, `krum`, `dnc`, `foolsgold`, `flame`, `lasa`.

## Evaluation

Benches: `advbench`, `directharm`, `expguardtest`, `pubmedqa`, `medQA`, `emrqa`, `cord19`.

Merge LoRA first for vLLM, then generate and judge:

```bash
python utils/merge_lora.py --base_model_path meta-llama/Llama-2-7b-chat-hf --lora_path <run_dir>/checkpoint-30

python evaluation/open_ended/gen_model_answer.py \
  --bench_name advbench --use_vllm --base_model_path <run_dir>/full-30 --gpu 0
# prints Saved to: .../<id>.json, use that <id> below(filename without .json)

python evaluation/open_ended/gen_judge_advbench.py \
  --bench_name advbench --model_answer <id> --judger rule
```

For `--use_vllm` and `full-30`, `<id>` is typically `<run_dir_basename>_30_vllm_chat_greedy`.

See `evaluation/open_ended/README.md`.

Code based on [https://github.com/dmqx/Safe-FedLLM](https://github.com/dmqx/Safe-FedLLM) and [https://github.com/19dx/FedLLM-Attack](https://github.com/19dx/FedLLM-Attack)