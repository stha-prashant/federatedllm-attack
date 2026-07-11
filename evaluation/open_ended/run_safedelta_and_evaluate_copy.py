# 1. Configuration
from pathlib import Path
import os, json, math, re, shutil, itertools, time
import torch
from typing import List, Dict, Any
import pdb
from wandb_utils import resolve_one_run


from pathlib import Path

def list_checkpoints_path(base_output_dir: str, args=None, safe_delta_criteria=False) -> List[Path]:
    """List all checkpoint directories in the given base output directory."""
    base_path = Path(base_output_dir)
    if not base_path.is_dir():
        print(f"[WARN] Base output dir {base_output_dir} does not exist or is not a directory.")
        return []
    # checkpoint_dirs = [p for p in base_path.glob('checkpoint*') if p.is_dir()]
    # checkpoint_dirs = [p for p in checkpoint_dirs if 'alpha' not in str(p.name)]
    checkpoint_dirs = [p for p in base_path.glob('full*') if p.is_dir()]
    checkpoint_dirs = [p for p in checkpoint_dirs if 'Safe' in str(p.name)]

    

    # if len(checkpoint_dirs_test) == 0:
    #     checkpoint_dirs = [p for p in checkpoint_dirs if 'alpha' in str(p.name)]
    # else:
    #     checkpoint_dirs = checkpoint_dirs_test
    if not args.safe_lora_original:
        checkpoint_dirs = [p for p in checkpoint_dirs if 'safe' not in str(p.name)]
    
    # checkpoint_dirs = sorted(checkpoint_dirs,
    #     key=lambda p: int(p.name.split('-')[-1].split('_')[0])
    # )


    
    return checkpoint_dirs




datasets = {'mtbench': './data/mtbench/questions.jsonl', 'advbench': './data/advbench/advbench.csv'}
import subprocess
TESTVLLM_LIB_PATH = "/home/ps9044/miniforge3/envs/testvllm/lib"
def generate_all_responses(runs_dict, ds='mtbench', eval_list=None, gpus=[0], args=None):
    processes = []
    gpu_id = 0
    original_eval_list = eval_list
    if args.safe_lora_original:
        for item in runs_dict:
            command = f"LD_LIBRARY_PATH={TESTVLLM_LIB_PATH} conda run -n testvllm python gen_model_answer.py --gpu {gpus[gpu_id]}  --use_vllm --base_model_path {str(item['safelora_original_saved_path']).replace('checkpoint', 'full')} --bench_name {ds}"
            print("Generating safelora original checkpoint: ", item["safelora_original_saved_path"])
            print(command)
            os.system(command)
    else:
        for item in runs_dict:
            if args.safe_delta_original:
                checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args, safe_delta_criteria=True)
            else:
                checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args)
            print(item['run_id'])
            # print(checkpoint_dirs)
            
            if original_eval_list is None:
                eval_list = ['50']
                if '100' in checkpoint_dirs[-1].name:
                    eval_list = ['100']
            
            # print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")
            for checkpoint_dir in checkpoint_dirs:
                # if checkpoint_dir.name.split('-')[-1].split('_')[0] in eval_list:

                gpu_id = (gpu_id + 1) % len(gpus)
                # print("Running generation file")
                if ds == 'mtbench':
                    command = f'python gen_model_answer_mt.py --gpu {gpus[gpu_id]} --base_model_path {item["model_name_or_path"]} --lora_path {checkpoint_dir}  --template {item["template"]}'
                    print("MTBENCH COMMAND: ", command)                
                else:
                    # command = f'python gen_model_answer.py --gpu {gpus[gpu_id]} --base_model_path {item["model_name_or_path"]} --lora_path {checkpoint_dir} --bench_name {ds}'
                    # if args.safe_delta_original:
                    #     checkpoint_dir  = str(checkpoint_dir)+f'SafeDeltasoriginal{args.safe_delta_thrs}'
                    command = f"LD_LIBRARY_PATH={TESTVLLM_LIB_PATH} conda run -n testvllm python gen_model_answer.py --gpu {gpus[gpu_id]}  --use_vllm --base_model_path {str(checkpoint_dir).replace('checkpoint', 'full')} --bench_name {ds}"

                print(command)
                # input("Press Enter to continue...")
                processes.append(subprocess.Popen(command, shell=True))
                print("-----------Num Processes--------------: ", len(processes))
                if len(processes) >= len(gpus):
                    for p in processes:
                        p.wait()
                    processes = []

def judge_all_responses_mt(runs_dict, ds='mtbench', eval_list = None, args=None):
    model_list = []

    for item in runs_dict:
        checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args)
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
                if checkpoint_dir.name.split('-')[-1].split('_')[0] in eval_list:
                    # model_list.append(f'{checkpoint_dir}/{ds}.jsonl')
                    model_list.append(get_save_path(checkpoint_dir, ds='mtbench'))

            else:
                print("No eval_list provided")
    joined_models = ' '.join(model_list)
    print(model_list)
    os.system(f'python gen_judge_mtbench.py --judge_model gpt-4o-mini --model_list {joined_models} --parallel 1')


def get_save_path(checkpoint_dir, ds='advbench', args=None):
    pre_str, checkpoint_str = os.path.split(checkpoint_dir)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
    result_path = f"./data/{ds}/model_answer/{model_name}.json"
    if ds == 'mtbench':
        return model_name
    else:
        return model_name + '_vllm_chat_greedy'
    # return result_path
    # return model_name + '_2'
    # return model_name + '_vllm'


def judge_all_responses(runs_dict, ds='advbench', eval_list=None, args=None):

    keyword = ''
    if args.safe_delta_original:
        args.safe_delta_thrs = str(args.safe_delta_thrs).replace('.', 'p')
        keyword = f'SafeDeltasoriginal{args.safe_delta_thrs}' 


    if args.safe_lora_original:
        for item in runs_dict:
            command = f'python gen_judge_advbench.py --judger rule --model_answer {get_save_path(item["safelora_original_saved_path"], ds)} --bench_name {ds} --round 30 --wandb_id {item["run_id"]}'
            print("Judging run id ", item["run_id"], "ds: ", ds)
            os.system(command)
    

    else:
        processes = []
        original_eval_list = eval_list
        for item in runs_dict:
            if args.safe_delta_original:
                checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args, safe_delta_criteria=True)
            else:
                checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args)
            # print(checkpoint_dirs)
            if original_eval_list is None:
                eval_list = ['100'] if '100' in checkpoint_dirs[-1].name else ['50']
            print("\n\n------------------------------------------------------------------------------------------------")
            print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")
            print("Judging run id ", item["run_id"])

            for checkpoint_dir in checkpoint_dirs:
                # if checkpoint_dir.name.split('-')[-1].split('_')[0] in eval_list:
                if True:
                    checkpoint_int = checkpoint_dir.name.split('-')[-1]
                    # if args.safe_delta_original:
                    #     checkpoint_dir  = str(checkpoint_dir)+f'SafeDeltasoriginal{args.safe_delta_thrs}'
                    print("Running judge file: ", str(checkpoint_dir).split('/')[-1])
                    os.system(f'python gen_judge_advbench.py --judger rule --model_answer {get_save_path(checkpoint_dir, ds)} --bench_name {ds} --round {checkpoint_int} --wandb_id {item["run_id"]}')
                    # print("Finished judging checkpoint: ", str(checkpoint_dir).split('/')[-1])

from copy import deepcopy

def merge_all_checkpoints(runs_dict, eval_list=None, args=None):
    if args.safe_lora_original:
        for item in runs_dict:
            # if path does not exist
            if not os.path.exists(str(item["safelora_original_saved_path"]).replace("checkpoint", "full")):
                command = f'python ../../utils/merge_lora.py --lora_path {item["safelora_original_saved_path"]} --base_model_path {item["model_name_or_path"]}'
                print("Merging safelora original checkpoint: ", item["safelora_original_saved_path"])
                print(command)
                os.system(command)
    else:
        print("Eval list: ", eval_list)
        print(list_checkpoints_path(runs_dict[0]["base_output_dir"], args=args))
        for item in runs_dict:
            checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args)

            print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")
            if eval_list is None:
                run_eval_list = [checkpoint_dir.name.split('-')[-1] for checkpoint_dir in checkpoint_dirs]
            else:
                run_eval_list = eval_list
            for checkpoint_dir in checkpoint_dirs:
                if checkpoint_dir.name.split('-')[-1].split('_')[0] in run_eval_list:
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

def run_safedelta(RUNS_TO_PROCESS, eval_list=None, args=None):
    for item in RUNS_TO_PROCESS:
        checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args)
        print(f"Found {len(checkpoint_dirs)} checkpoints in {item['base_output_dir']}")

        if eval_list is None:
            run_eval_list = [checkpoint_dir.name.split('-')[-1] for checkpoint_dir in checkpoint_dirs]
        else:
            run_eval_list = eval_list

        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.name.split('-')[-1].split('_')[0] in run_eval_list:
                print("Generating SafeDelta model for checkpoint: ", str(checkpoint_dir).split('/')[-1])
                model_name_align = item["model_name_or_path"]
                model_name_ft = str(checkpoint_dir).replace("checkpoint", "full")
                command = f'python ../../SafeDelta/llama2/run_safedelta.py --model_name_align {model_name_align} --model_name_ft {model_name_ft} --scale {args.safe_delta_thrs}'
                print(command)
                os.system(command)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # list of datasets to evaluate
    parser.add_argument('--datasets', type=str, nargs='+', default=['advbench', ], help='List of datasets to evaluate')
    parser.add_argument('--eval_list', type=str, nargs='+', default=['30'], help='List of checkpoint ids to evaluate')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='List of GPU ids to use for generation')
    parser.add_argument('--run_ids', type=str, nargs='+', default=None, help='List of W&B run ids or run paths to process')
    parser.add_argument('--safe_lora_original', action='store_true', help='different paths according to this flag')
    parser.add_argument('--safe_delta_original', action='store_true', help='whether to generate SafeDelta model for original safelora checkpoint')
    parser.add_argument('--safe_delta_thrs', type=float, default=0.1, help='the s value for SafeDelta when safe_delta_original flag is set')
    args = parser.parse_args()



    # extract info from run_ids
    WANDB_RUN_IDS = [str(x) for x in args.run_ids]
    RUNS_TO_PROCESS = []
    for rid in WANDB_RUN_IDS:
        RUNS_TO_PROCESS.append(resolve_one_run(rid, args=args))


    for eval_item in args.eval_list:
        current_eval_list = [eval_item]

    
        # # merge checkpoints
        # merge_all_checkpoints(RUNS_TO_PROCESS, eval_list=current_eval_list, args=args)
        print("Merged checkpoint-------------------------------------------------\n\n")
      



        # # generate safe_delta model
        if args.safe_delta_original:
            # run_safedelta(RUNS_TO_PROCESS, eval_list=current_eval_list, args=args)
            print("Generated SafeDelta models -------------------------------------------------\n\n")
            args.safe_delta_thrs = str(args.safe_delta_thrs).replace('.', 'p')
            # exit()

        #   for each dataset, generate
        for ds in args.datasets:
            generate_all_responses(RUNS_TO_PROCESS, ds=ds, gpus=args.gpus, eval_list=current_eval_list, args=args)
            pass
        print("Generated responses -------------------------------------------------\n\n")


        # # for each dataset, judge
        for ds in args.datasets:
            if ds == 'mtbench':
                judge_all_responses_mt(RUNS_TO_PROCESS, ds=ds, eval_list=current_eval_list)
                show_all_responses_mt(RUNS_TO_PROCESS, eval_list=current_eval_list)
            else:
                judge_all_responses(RUNS_TO_PROCESS, ds=ds, eval_list=current_eval_list, args=args)

        
        # show results
        # delete merged full models to save space
        for item in RUNS_TO_PROCESS:
            if args.safe_lora_original:
                shutil.rmtree(str(item["safelora_original_saved_path"]).replace("checkpoint", "full"), ignore_errors=True)
            else:
                checkpoint_dirs = list_checkpoints_path(item["base_output_dir"], args=args, safe_delta_criteria=args.safe_delta_original)
                for checkpoint_dir in checkpoint_dirs:
                    if checkpoint_dir.name.split('-')[-1].split('_')[0] in current_eval_list:
                        full_model_path = str(checkpoint_dir).replace("checkpoint", "full")
                        if args.safe_delta_original:
                            full_model_path = str(checkpoint_dir)+f'SafeDeltasoriginal{args.safe_delta_thrs}'
                            print(full_model_path)
                        
                        print("Deleting full model at: ", full_model_path)
                        shutil.rmtree(full_model_path, ignore_errors=True)

    




