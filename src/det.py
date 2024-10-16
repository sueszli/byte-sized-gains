import os
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.benchmark import Timer
from tqdm import tqdm
from transformers import DetrFeatureExtractor, DetrForObjectDetection

from datasets import load_dataset
from utils import get_device, set_env

set_env(seed=42)

classes_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(classes_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


def main(args: dict):
    device = get_device(disable_mps=False)
    model_id = "facebook/detr-resnet-101-dc5"
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_id, cache_dir=weights_path, local_files_only=True)
    model = DetrForObjectDetection.from_pretrained(model_id, cache_dir=weights_path, local_files_only=True)
    model.eval()
    model.to(device)

    testset = load_dataset("rafaelpadilla/coco2017", split="val", streaming=False, cache_dir=dataset_path)  # can't map(toTensor()) in-memory for batching

    for sample in tqdm(testset):
        metrics = {}

        # ground truth
        image: Image.Image = sample["image"]
        image_id: int = sample["image_id"]
        true_bboxes: dict = sample["objects"]  # {"bbox": [[x, y, w, h]], "label": [int]}

        # preprocessing
        image = image.convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # benchmark inference
        def inference():
            with torch.inference_mode():
                outputs = model(**inputs)
            return outputs

        timer = Timer(
            stmt="inference()",
            globals={"inference": inference, "model": model, "inputs": inputs, "torch": torch},
        )
        measurement = timer.blocked_autorange()
        metrics["inference_time"] = measurement.mean

        # inference
        with torch.inference_mode():
            outputs = model(**inputs)

        # postprocessing
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=args.threshold)[0]
        results = {k: v.cpu() for k, v in results.items()}
        results["labels_names"] = [model.config.id2label[elem] for elem in [elem.cpu().item() for elem in results["labels"]]]

        # evaluation
        # print(results["scores"])
        # print(results["boxes"])
        # print(results["labels"])

        print(results["labels_names"])

    # https://ai.google.dev/edge/litert/models/convert_pytorch


if __name__ == "__main__":
    args = SimpleNamespace(
        threshold=0.9,
    )
    main(args)
