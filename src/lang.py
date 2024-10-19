from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from datasets import load_dataset
import time
import csv
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bitsandbytes.nn import LinearFP4, Linear8bitLt
import numpy as np

from utils import get_device
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)



dataset = load_dataset("cimec/lambada", split="test", streaming=False, cache_dir=dataset_path)

def quantize_and_save(bits):
    device = get_device(disable_mps=False)
    model_id = "HuggingFaceTB/SmolLM-135M"

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=weights_path)

    # get calibration data
    def prepare_dataset(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(prepare_dataset, batched=True, remove_columns=dataset.column_names)
    calibration_data = [item['input_ids'] for item in tokenized_dataset]

    # config quantization
    gptq_config = GPTQConfig(
        bits=bits,
        dataset=calibration_data[:100],
        tokenizer=tokenizer,
        block_name_to_quantize="model.layers",
        model_seqlen=2048,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config, cache_dir=weights_path).to(device)
    
    model.save_pretrained(f"{model_id}-{bits}bit-gptq")
    tokenizer.save_pretrained(f"{model_id}-{bits}bit-gptq")


def main(args):
    device = get_device(disable_mps=False)
    model_id = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=weights_path)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=weights_path).to(device)
    model.eval()

    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

    
    
    

    
    


    # def predict(text):
    #     inputs = tokenizer(text, return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         outputs = model.generate(**inputs, max_new_tokens=1)
    #     return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # correct = 0
    # total = 0

    # for example in tqdm(testset):
    #     context = example['text'][:-1]  # drop last word
    #     target = example['text'][-1]  # set last word as target
        
    #     prediction = predict(context)
        
    #     if prediction.strip().endswith(target):
    #         correct += 1
    #     total += 1

    #     if total % 100 == 0:
    #         print(f"processed {total} examples. accuracy: {correct/total:.2%}")

    # print(f"Final Accuracy: {correct/total:.2%}")



    # quantize weights only
    configs = ["int8", "int4", "int2"]






if __name__ == "__main__":
    for bits in [2, 4, 8]:
        print(f"quantizing to {bits} bits")
        quantize_and_save(bits)


    args = SimpleNamespace(
        sample_size=1500,
    )
    # main(args)
