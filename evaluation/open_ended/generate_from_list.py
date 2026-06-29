# 1. Configuration
from pathlib import Path
import os, json, math, re, shutil, itertools, time
from PIL.Image import item
import torch
from typing import List, Dict, Any
from wandb_utils import resolve_one_run


from pathlib import Path


datasets = {'mtbench': '/home/ps9044/RPA/fedllm-attack/evaluation/open_ended/data/mtbench/questions.jsonl', 'advbench': '/home/ps9044/RPA/fedllm-attack/evaluation/open_ended/data/advbench/advbench.csv'}
import subprocess
def generate_all_responses(runs_dict, ds='mtbench', eval_list=None, gpus=[0], args=None):
    processes = []
    gpu_id = 0
    original_eval_list = eval_list

    # checkpoint_dirs = ['meta-llama/Llama-2-7b-chat-hf']
    checkpoint_dirs = list_checkpoints_path('/shared/rc/llm-degredation')
    # print(checkpoint_dirs)
    
    # print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")
    for checkpoint_dir in checkpoint_dirs:
        if 'full' in checkpoint_dir:
            gpu_id = (gpu_id + 1) % len(gpus)
            # print("Running generation file")
                # command = f'python gen_model_answer.py --gpu {gpus[gpu_id]} --base_model_path {item["model_name_or_path"]} --lora_path {checkpoint_dir} --bench_name {ds}'
            command = f"conda run -n testvllm python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/gen_model_answer.py --gpu {gpus[gpu_id]}  --use_vllm --base_model_path {str(checkpoint_dir)} --bench_name {ds}"

            print(command)
            # input("Press Enter to continue...")
            processes.append(subprocess.Popen(command, shell=True))
            print("-----------Num Processes--------------: ", len(processes))
            if len(processes) >= len(gpus):
                for p in processes:
                    p.wait()
                processes = []


def get_save_path(checkpoint_dir, ds='advbench', args=None):
    pre_str, last_str = os.path.split(checkpoint_dir)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str  
    return model_name + '_vllm_chat_greedy'
    # return result_path
    # return model_name + '_2'
    # return model_name + '_vllm'


def list_checkpoints_path(checkpoint_dir):
    checkpoint_dirs = []
    for item in os.listdir(checkpoint_dir):
        if ('checkpoint' in item or 'full' in item) and os.path.isdir(os.path.join(checkpoint_dir, item)) and 'rank' in item and 'normalized' in item:
            checkpoint_dirs.append(os.path.join(checkpoint_dir, item))
    return checkpoint_dirs

def merge_all_checkpoints(checkpoint_dir, eval_list=None):
    checkpoint_dirs = list_checkpoints_path(checkpoint_dir)
    print(checkpoint_dirs)
   
    for checkpoint_dir in checkpoint_dirs:
        if 'full' not in checkpoint_dir:
            print("Merging checkpoint: ", str(checkpoint_dir).split('/')[-1])
            command = f'python ../../utils/merge_lora.py --lora_path {checkpoint_dir} --base_model_path meta-llama/Llama-2-7b-chat-hf'
            print(command)
            os.system(command)


def judge_all_responses(runs_dict, ds='advbench', eval_list=None, args=None):

    # checkpoint_dirs = ['meta-llama/Llama-2-7b-chat-hf']
    checkpoint_dirs = list_checkpoints_path('/shared/rc/llm-degredation')
    # print(checkpoint_dirs)
    
    for checkpoint_dir in checkpoint_dirs:
        if 'full' in checkpoint_dir:
            keyword=get_save_path(checkpoint_dir, ds)
            print("Running judge file: ", str(checkpoint_dir).split('/')[-1])
            os.system(f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/gen_judge_advbench.py --judger rule --model_answer {get_save_path(checkpoint_dir, ds)} --bench_name {ds} --round {0} --wandb_id v376gich --keyword {keyword}')
            print("Finished judging checkpoint: ", str(checkpoint_dir).split('/')[-1])

from copy import deepcopy

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # list of datasets to evaluate
    parser.add_argument('--datasets', type=str, nargs='+', default=['advbench', ], help='List of datasets to evaluate')
    parser.add_argument('--eval_list', type=str, nargs='+', default=['30'], help='List of checkpoint ids to evaluate')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of GPU ids to use for generation')
    parser.add_argument('--run_ids', type=str, nargs='+', default=None, help='List of W&B run ids or run paths to process')
    parser.add_argument('--safe_lora_original_minimal', action='store_true', help='different paths according to this flag')
    
    parser.add_argument('--safe_lora_original', action='store_true', help='different paths according to this flag')
    args = parser.parse_args()

    # extract info from run_ids
    RUNS_TO_PROCESS = []
    for eval_item in args.eval_list:
        current_eval_list = [eval_item]

    
        # merge checkpoints
        merge_all_checkpoints('/shared/rc/llm-degredation', eval_list=current_eval_list)
        print("Merged checkpoint-------------------------------------------------\n\n")
        # for each dataset, generate
        for ds in args.datasets:
            generate_all_responses(RUNS_TO_PROCESS, ds=ds, gpus=args.gpus, eval_list=current_eval_list, args=args)
            pass
        print("Generated responses -------------------------------------------------\n\n")

        # # for each dataset, judge
        for ds in args.datasets:
            judge_all_responses(RUNS_TO_PROCESS, ds=ds, eval_list=current_eval_list, args=args)



