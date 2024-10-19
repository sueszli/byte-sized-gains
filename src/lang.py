import os
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from datasets import load_dataset
from utils import get_device

output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)

assert get_device(disable_mps=False) == "cuda", "model quantization requires a GPU"

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
        quantize_and_save(bits)

    bits = 8

    modelpath = weights_path / f"quantized-smollm135m-{bits}bits"
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model.eval()

    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    args = SimpleNamespace(
        sample_size=1500,
    )
    main(args)
