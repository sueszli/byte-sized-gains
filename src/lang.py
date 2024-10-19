import csv
import gc
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from datasets import load_dataset
from utils import get_device

assert get_device(disable_mps=False) == "cuda", "model quantization requires a GPU"
torch.cuda.set_device(0)

output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


dataset = load_dataset("cimec/lambada", split="test", streaming=False, cache_dir=dataset_path)


def quantize_and_save(bits):
    modelpath = weights_path / f"quantized-smollm135m-{bits}bits"
    if os.path.exists(modelpath):
        print(f"{modelpath} already exists, skipping")
        return

    model_id = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=weights_path)

    gptq_config = GPTQConfig(
        bits=bits,
        group_size=64,
        dataset=dataset.select(range(100))["text"],
        tokenizer=tokenizer,
        block_name_to_quantize="model.layers",
        model_seqlen=2048,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config, cache_dir=weights_path, device_map="auto")

    model.save_pretrained(modelpath)
    tokenizer.save_pretrained(modelpath)


def main(args):
    for bits in [2, 4, 8]:
        print(f"quantizing model to {bits}-bit")
        quantize_and_save(bits)

        print(f"benchmarking {bits}-bit model")
        modelpath = weights_path / f"quantized-smollm135m-{bits}bits"
        model = AutoModelForCausalLM.from_pretrained(modelpath)
        tokenizer = AutoTokenizer.from_pretrained(modelpath)
        model.eval()
        model.to("cuda")

        correct_1 = 0
        correct_k = 0
        total = 0
        total_tokens = 0
        start_time = time.time()

        for i, example in tqdm(enumerate(dataset), total=args.sample_size, desc=f"{bits}-bit"):
            if i >= args.sample_size:
                break

            context = example["text"][:-1]  # drop last word
            target = example["text"][-1]  # set it as target

            inputs = tokenizer(context, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            total_tokens += input_length + 1  # input tokens + 1 generated token

            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits[:, -1, :]
            top_5_predictions = torch.topk(logits, 5).indices[0]
            target_id = tokenizer.encode(target, add_special_tokens=False)[0]
            if target_id == top_5_predictions[0]:
                correct_1 += 1
            if target_id in top_5_predictions:
                correct_k += 1
            total += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        stats = {
            "bits": bits,
            "top_1_accuracy": correct_1 / total,
            "top_5_accuracy": correct_k / total,
            "tokens_per_second": total_tokens / elapsed_time,
            "total_samples": total,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time,
            "memory_footprint_mb": model.get_memory_footprint() / 1e6,
        }
        with open(output_path / f"lang.csv", "a") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if i == 0:
                writer.writeheader()
            writer.writerow(stats)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = SimpleNamespace(
        sample_size=5000,
    )
    main(args)
