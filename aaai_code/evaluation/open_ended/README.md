# Evaluation

Supported benches: `advbench`, `directharm`, `expguardtest`, `pubmedqa`, `medQA`, `emrqa`, `cord19`.

Merge a FL checkpoint first (e.g. `checkpoint-30` → `full-30`) if you use vLLM:

```bash
python utils/merge_lora.py --base_model_path meta-llama/Llama-2-7b-chat-hf --lora_path <run_dir>/checkpoint-30
```

## 1) Generate answers

```bash
python evaluation/open_ended/gen_model_answer.py \
  --bench_name advbench --use_vllm \
  --base_model_path <run_dir>/full-30 --gpu 0
```

This writes:

`evaluation/open_ended/data/<bench>/model_answer/<id>.json`

and prints `Saved to: ...` / `model name: ...`.

**How `<id>` is formed**

- From a merged full model `<run_dir>/full-30`:
  - `<id> = <run_dir_basename>_30_vllm_chat_greedy` when using `--use_vllm`
  - `<id> = <run_dir_basename>_30` without vLLM
- Example: if the run folder is `Llama-2-7b-chat-hf_..._ours_..._20260701`, then
  `--model_answer Llama-2-7b-chat-hf_..._ours_..._20260701_30_vllm_chat_greedy`

You can also copy `<id>` from the printed path (filename without `.json`).

## 2) Judge / score

```bash
python evaluation/open_ended/gen_judge_advbench.py \
  --bench_name advbench \
  --model_answer <id> \
  --judger rule
```

Judgments are saved to:

`evaluation/open_ended/data/<bench>/model_judgment/rule_<id>.json`

Safety benches (`advbench`, `directharm`, `expguardtest`) use refusal-prefix + ExpGuard scoring. Utility benches report task accuracy.
