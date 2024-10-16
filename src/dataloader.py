import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
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
    model_id = "facebook/detr-resnet-101-dc5"  # largest model
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_id, cache_dir=weights_path, local_files_only=True)  # maybe also threshold arg
    model = DetrForObjectDetection.from_pretrained(model_id, cache_dir=weights_path, local_files_only=True)
    model.eval()
    model.to(device)

    coco_classes = json.load(open(classes_path / "coco_classes.json", "r"))
    testset = load_dataset("rafaelpadilla/coco2017", split="val", streaming=False, cache_dir=dataset_path)  # can't .map(toTensor()) in memory for batching

    for sample in tqdm(testset):
        image: Image.Image = sample["image"]
        image_id: int = sample["image_id"]
        true_bboxes: dict = sample["objects"]

        image = image.convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = feature_extractor.post_process_object_detection(outputs, args.threshold, target_sizes=target_sizes)

        # outputs = {k: v.cpu() for k, v in outputs.items()}
        # logits = outputs["logits"]
        # bboxes = outputs["pred_boxes"]
        print(len(results))


if __name__ == "__main__":
    args = SimpleNamespace(
        threshold=0.5,
    )
    main(args)


# https://ai.google.dev/edge/litert/models/convert_pytorch

# with torch.no_grad():
#     outputs = model(**inputs)

# target_sizes = torch.tensor([img.size[::-1]]).to(device)
# results = image_processor.post_process_object_detection(outputs, threshold, target_sizes=target_sizes)[0]

# results["boxes"] = [elem.cpu().tolist() for elem in results["boxes"]]
# results["scores"] = [elem.cpu().item() for elem in results["scores"]]
# results["labels"] = [model.config.id2label[elem.cpu().item()] for elem in results["labels"]]

# boxes = results["boxes"]
# scores = results["scores"]
# labels = results["labels"]

# # Assertions for type checking
# assert all(isinstance(label, str) for label in labels)
# assert all(isinstance(score, float) for score in scores)
# assert all(isinstance(box, list) and len(box) == 4 for box in boxes)
# assert all(all(isinstance(coord, float) for coord in box) for box in boxes)
