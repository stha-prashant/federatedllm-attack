import torch
from tqdm import tqdm

def query(data, model, tokenizer):
    model.eval()
    instruction = data["instruction"]
    input = data["input"]
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
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


def compute_sst2_accuracy(model, tokenizer):
    from datasets import load_dataset
    dataset = load_dataset("stanfordnlp/sst2", split='validation')
    input_data_lst = []
    index = 0
    for example in dataset:
        if  index<100 :
            instance = {}
            instance["instruction"] = "Analyze the sentiment of the input, and respond only positive or negative"
            instance["input"] = example["sentence"]
            instance["label"] = example["label"]
            input_data_lst += [instance]
            index+=1

    pred_lst = []
    for data in tqdm(input_data_lst):
        pred = query(data, model, tokenizer)
        pred_lst.append(pred)


    correct = 0
    total = 0
    output_lst = []
    for input_data, pred in zip(input_data_lst, pred_lst):
        input_data['output'] = pred
        if input_data ["label"]:
            label1 = "positive"
            label2 = "Positive"
        else:
            label1 = "negative"
            label2 = "Negative"
            
        if label1 in pred or label2 in pred:
            correct +=1 
            input_data["correct"] ="True"
        else:
            input_data["correct"] ="False"
        total += 1
        output_lst.append(input_data)
    accuracy = correct / total
    output_lst.append("Overall Accuracy: {:.2f}%".format(accuracy * 100))
    print("SST-2 Accuracy: {:.2f}%".format(accuracy * 100))
    return accuracy, output_lst