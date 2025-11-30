
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nFirst think step by step and then answer the final number.\n"
import torch
from tqdm import tqdm

def extract_answer_number(sentence: str) -> float:
    import re
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def gsm8k_format(example):
    example['instruction'] = f"{example['question']}{QUESTION_PROMPT}"
    example['response'] = f"{example['answer']}".replace("#### ", ANSWER_PROMPT)
    return example


def query(data, model, tokenizer):
    model.eval()
    instruction = data["instruction"]
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    model.train()
    return res



def compute_gsm8k_accuracy(model, tokenizer):
    from datasets import load_dataset
    dataset = load_dataset("HongzheBi/gsm8k", split='test')
    input_data_lst = []
    index = 0
    for data in dataset:
        if index < 100:
            item = {}
            item['instruction'] = f"{data['question']}{QUESTION_PROMPT}"
            item['response'] = f"{data['answer']}".replace("#### ", ANSWER_PROMPT)
            input_data_lst += [item]
            index += 1

    pred_lst = []
    for data in tqdm(input_data_lst):
        pred = query(data, model, tokenizer)
        pred_lst.append(pred)


    correct = 0
    total = 0
    output_lst = []
    for input_data, pred in zip(input_data_lst, pred_lst):
        answer_ground_truth = extract_answer_number(input_data['response'])
        answer = extract_answer_number(pred)
        input_data['output'] = pred

        if answer == answer_ground_truth:
            input_data['correct'] = 'True'
            correct += 1
        else:
            input_data['correct'] = 'False'
        total += 1
        output_lst.append(input_data)


    accuracy = correct / total
    output_lst.append("Overall Accuracy: {:.2f}%".format(accuracy * 100))
    print("GSM8K Accuracy: {:.2f}%".format(accuracy * 100))

    return accuracy, output_lst