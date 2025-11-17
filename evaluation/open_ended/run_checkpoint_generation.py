# 1. Configuration
from pathlib import Path
import os, json, math, re, shutil, itertools, time
import torch
from typing import List, Dict, Any

USE_NEPTUNE = True
# NEPTUNE_RUN_IDS = [f'FED-{x}' for x in [68, 77]] #base wildchat and postfine of that
# NEPTUNE_RUN_IDS = [f'FED-{x}' for x in [51, 136, 10, 124, 62, 142, 137]] #base wildchat and postfine of that
# NEPTUNE_RUN_IDS = [f'FED-{x}' for x in [68]] #base wildchat and postfine of that
# NEPTUNE_RUN_IDS = [f'FED-{x}' for x in [10]] #base wildchat and postfine of that

NEPTUNE_RUN_IDS = [f'FED-{x}' for x in [178]]

NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT", "fedllm/fedllm")

RUNS_TO_PROCESS = []  # list of dicts: {"run_id": str, "base_output_dir": str, "model_name_or_path": str|None}

if USE_NEPTUNE and NEPTUNE_RUN_IDS:
    import neptune
    token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDNlMTFjYi0wMzQ4LTRmMDUtOTk4NC0wZjBlOGU5NGExMmYifQ=='

    def resolve_one_run(run_id: str):
        print("here")
        run = neptune.init_run(project=NEPTUNE_PROJECT, with_id=run_id, api_token=token, mode="read-only")
        print("done")
        resolved = None
        for field in [
            "parameters/total_output_dir",
            "parameters/script_args/output_dir",
        ]:
            try:
                v = run[field].fetch()
                if isinstance(v, str) and v:
                    resolved = v
                    break
            except Exception:
                pass
        try:
            model_name_or_path = run["parameters/script_args/model_name_or_path"].fetch()
            num_rounds = run['parameters/fed_args/num_rounds'].fetch()
            sample_clients = run['parameters/fed_args/sample_clients'].fetch()
            template = run['parameters/script_args/template'].fetch()
        except Exception:
            model_name_or_path = None

        # Fallback: best-effort local guess by run id name
        if not resolved:
            from pathlib import Path
            def _guess_dir_from_run_id(rid: str) -> str | None:
                candidates = [Path("./outputs"), Path(".")]
                for root in candidates:
                    if (root / rid).is_dir() and (root / rid / "lora_updates").is_dir():
                        return str(root / rid)
                    for p in root.glob(f"**/{rid}"):
                        if p.is_dir() and (p / "lora_updates").is_dir():
                            return str(p)
                return None
            guess = _guess_dir_from_run_id(run_id)
            if guess:
                resolved = guess
                print(f"[Neptune] Guessed BASE_OUTPUT_DIR for {run_id}: {resolved}")
        print(resolved, model_name_or_path, num_rounds, sample_clients)
        # if resolved:
        RUNS_TO_PROCESS.append({
            "run_id": run_id,
            "base_output_dir": resolved,
            "model_name_or_path": model_name_or_path,
            "num_rounds": num_rounds,
            "sample_clients": sample_clients,
            "template": template
        })
        # else:
        #     print(f"[WARN] Could not resolve output_dir for run {run_id}; skipping.")
        # run.stop()


    for rid in NEPTUNE_RUN_IDS:
        resolve_one_run(rid)


from pathlib import Path

def list_checkpoints_path(base_output_dir: str) -> List[Path]:
    """List all checkpoint directories in the given base output directory."""
    base_path = Path(base_output_dir)
    if not base_path.is_dir():
        print(f"[WARN] Base output dir {base_output_dir} does not exist or is not a directory.")
        return []
    checkpoint_dirs = sorted([p for p in base_path.glob('checkpoint-*') if p.is_dir()],
        key=lambda p: int(p.name.split('-')[-1])
    )
    return checkpoint_dirs




datasets = {'mtbench': './data/mtbench/questions.jsonl', 'advbench': './data/advbench/advbench.csv'}
import subprocess
def generate_all_responses(runs_dict, ds='mtbench', eval_list=None, gpus=[0]):
    processes = []
    gpu_id = 0
    original_eval_list = eval_list

    for item in runs_dict:
        checkpoint_dirs = list_checkpoints_path(item["base_output_dir"])
        print(item['run_id'])
        # print(checkpoint_dirs)
        
        if original_eval_list is None:
            eval_list = ['50']
            if '100' in checkpoint_dirs[-1].name:
                eval_list = ['100']
        
        # print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.name.split('-')[-1] in eval_list:

                gpu_id = (gpu_id + 1) % len(gpus)
                # print("Running generation file")
                if ds == 'mtbench':
                    command = f'python gen_model_answer_mt.py --gpu {gpus[gpu_id]} --base_model_path {item["model_name_or_path"]} --lora_path {checkpoint_dir}  --template {item["template"]}'
                    print("MTBENCH COMMAND: ", command)                
                else:
                    # command = f'python gen_model_answer.py --gpu {gpus[gpu_id]} --base_model_path {item["model_name_or_path"]} --lora_path {checkpoint_dir} --bench_name {ds}'
                    command = f"python gen_model_answer.py --gpu {gpus[gpu_id]}  --use_vllm --base_model_path {str(checkpoint_dir).replace('checkpoint', 'full')} --bench_name {ds}"

                print(command)
                # input("Press Enter to continue...")
                processes.append(subprocess.Popen(command, shell=True))
                print("-----------Num Processes--------------: ", len(processes))
                if len(processes) >= len(gpus):
                    for p in processes:
                        p.wait()
                    processes = []

def judge_all_responses_mt(runs_dict, ds='mtbench', eval_list = None):
    model_list = []

    for item in runs_dict:
        checkpoint_dirs = list_checkpoints_path(item["base_output_dir"])
        print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")

        # found_100 = False
        # for checkpoint_dir in checkpoint_dirs:
        #     if eval_list is None and checkpoint_dir.name.split('-')[-1] in ['100']:
        #         model_list.append(f'{checkpoint_dir}/{ds}.jsonl')
        #         found_100 = True

        #     elif eval_list is not None and checkpoint_dir.name.split('-')[-1] in eval_list:
        #         model_list.append(f'{checkpoint_dir}/{ds}.jsonl')
        # if not found_100:
        #     for checkpoint_dir in checkpoint_dirs:
        #         if eval_list is None and checkpoint_dir.name.split('-')[-1] in ['50']:
        #             model_list.append(f'{checkpoint_dir}/{ds}.jsonl')
        #         elif eval_list is not None and checkpoint_dir.name.split('-')[-1] in eval_list:
        #             model_list.append(f'{checkpoint_dir}/{ds}.jsonl')

        for checkpoint_dir in checkpoint_dirs:
            if eval_list is not None:
                if checkpoint_dir.name.split('-')[-1] in eval_list:
                    # model_list.append(f'{checkpoint_dir}/{ds}.jsonl')
                    model_list.append(get_save_path(checkpoint_dir, ds='mtbench'))

            else:
                print("No eval_list provided")
    joined_models = ' '.join(model_list)
    print(model_list)
    os.system(f'python gen_judge_mtbench.py --judge_model gpt-4o-mini --model_list {joined_models} --parallel 1')


def get_save_path(checkpoint_dir, ds='advbench'):
    pre_str, checkpoint_str = os.path.split(checkpoint_dir)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
    result_path = f"./data/{ds}/model_answer/{model_name}.json"
    if ds == 'mtbench':
        return model_name
    else:
        return model_name + '_vllm'
    # return result_path
    # return model_name + '_2'
    # return model_name + '_vllm'


def judge_all_responses(runs_dict, ds='advbench', eval_list=None):
    processes = []
    original_eval_list = eval_list
    for item in runs_dict:
        checkpoint_dirs = list_checkpoints_path(item["base_output_dir"])
        # print(checkpoint_dirs)
        if original_eval_list is None:
            eval_list = ['100'] if '100' in checkpoint_dirs[-1].name else ['50']
        print("\n\n------------------------------------------------------------------------------------------------")
        print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")
        print("Judging run id ", item["run_id"])

        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.name.split('-')[-1] in eval_list:
            # if True:
                print("Running judge file: ", str(checkpoint_dir).split('/')[-1])
                os.system(f'python gen_judge_advbench.py --judger rule --model_answer {get_save_path(checkpoint_dir, ds)} --bench_name {ds}')

from copy import deepcopy

def merge_all_checkpoints(runs_dict, eval_list=None):
    for item in runs_dict:
        checkpoint_dirs = list_checkpoints_path(item["base_output_dir"])

        print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")
        if eval_list is None:
            run_eval_list = [checkpoint_dir.name.split('-')[-1] for checkpoint_dir in checkpoint_dirs]
        else:
            run_eval_list = eval_list
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.name.split('-')[-1] in run_eval_list:
                print("Merging checkpoint: ", str(checkpoint_dir).split('/')[-1])
                command = f'python ../../utils/merge_lora.py --lora_path {checkpoint_dir} --base_model_path {item["model_name_or_path"]}'
                print(command)
                os.system(command)

def show_all_responses_mt(runs_dict, eval_list=['50', '100']):
    model_list = []

    for item in runs_dict:
        checkpoint_dirs = list_checkpoints_path(item["base_output_dir"])
        print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")

        found_100 = False
        for checkpoint_dir in checkpoint_dirs:
        #     if checkpoint_dir.name.split('-')[-1] in ['100']:
        #         model_list.append(f'{checkpoint_dir}/mtbench.jsonl')
        #         found_100 = True
        # if not found_100:
        #     for checkpoint_dir in checkpoint_dirs:
        #         if checkpoint_dir.name.split('-')[-1] in ['50']:
        #             model_list.append(f'{checkpoint_dir}/mtbench.jsonl')
            # model_list.append(f'{checkpoint_dir}/mtbench.jsonl')
            model_list.append(get_save_path(checkpoint_dir, ds='mtbench'))
    joined_models = ' '.join(model_list)
    print(model_list)
    os.system(f'python show_results_mt.py --judge_model gpt-4o-mini --model_list {joined_models}')
print(len(RUNS_TO_PROCESS))
# merge_all_checkpoints(RUNS_TO_PROCESS, eval_list=['50'])
# merge_all_checkpoints(RUNS_TO_PROCESS[0:1], eval_list=['100'])
# merge_all_checkpoints(RUNS_TO_PROCESS[1:], eval_list=['10', '20', '30', '40', '50'])

generate_all_responses(RUNS_TO_PROCESS, 'advbench', gpus=[7], eval_list=['50'])
# judge_all_responses(RUNS_TO_PROCESS, 'advbench', eval_list=['50'])

# generate_all_responses(RUNS_TO_PROCESS[0:1], 'advbench', gpus=[4], eval_list=['100'])
# generate_all_responses(RUNS_TO_PROCESS[0:1], 'val_postfinetune', gpus=[4], eval_list=['100'])

# generate_all_responses(RUNS_TO_PROCESS[1:], 'advbench', gpus=[4, 5, 6], eval_list=['10', '20', '30', '40', '50'])
# generate_all_responses(RUNS_TO_PROCESS[1:], 'val_postfinetune', gpus=[4, 5, 6], eval_list=['10', '20', '30', '40', '50'])



# generate_all_responses(RUNS_TO_PROCESS[0:1], 'mtbench', gpus=[4], eval_list=['100'])
# generate_all_responses(RUNS_TO_PROCESS[1:], 'mtbench', gpus=[4, 6], eval_list=['10', '20', '30', '40', '50'])

# show_all_responses_mt(RUNS_TO_PROCESS[1:], eval_list=['10', '20', '30', '40', '50'])
# show_all_responses_mt(RUNS_TO_PROCESS[0:1], eval_list=['100'])

# judge_all_responses(RUNS_TO_PROCESS[0:1], 'advbench', eval_list=['100'])
# judge_all_responses(RUNS_TO_PROCESS[1:], 'advbench', eval_list=['10', '20', '30', '40', '50'])
# judge_all_responses(RUNS_TO_PROCESS[0:1], 'val_postfinetune', eval_list=['100'])
# judge_all_responses(RUNS_TO_PROCESS[1:], 'val_postfinetune', eval_list=['10', '20', '30', '40', '50'])

# judge_all_responses(RUNS_TO_PROCESS, 'advbench', eval_list=['20', '50'])
# generate_all_responses(RUNS_TO_PROCESS, 'val_postfinetune', gpus=[2], eval_list=['100'])
# judge_all_responses(RUNS_TO_PROCESS, 'val_postfinetune', eval_list=['100'])
# judge_all_responses(RUNS_TO_PROCESS, 'advbench', eval_list=['100'])


# judge_all_responses(RUNS_TO_PROCESS, 'val_postfinetune', eval_list=['20', '50'])

