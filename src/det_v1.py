"""
had to throw this away because `ai_edge_torch` has some weird cross dependency i couldn't resolve :(

see: https://github.com/google-ai-edge/ai-edge-torch?tab=readme-ov-file#update-ld_library_path-if-necessary
"""

import csv
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

output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


def compute_precision(pred_labels, pred_bboxes, true_labels, true_bboxes, iou_threshold):
    true_positives = 0
    false_positives = 0

    def compute_iou(box1, box2):
        # convert [x, y, w, h] to [x1, y1, x2, y2]
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

        inter_area = max(0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * max(0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    for pred_label, pred_bbox in zip(pred_labels, pred_bboxes):
        matched = False
        for true_label, true_bbox in zip(true_labels, true_bboxes):
            if pred_label == true_label and compute_iou(pred_bbox, true_bbox) >= iou_threshold:
                true_positives += 1
                matched = True
                break
        if not matched:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def main(args: dict):
    device = get_device(disable_mps=False)
    model_id = "facebook/detr-resnet-101-dc5"
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_id, cache_dir=weights_path, local_files_only=True)
    model = DetrForObjectDetection.from_pretrained(model_id, cache_dir=weights_path, local_files_only=True)
    model.eval()
    model.to(device)

    testset = load_dataset("rafaelpadilla/coco2017", split="val", streaming=False, cache_dir=dataset_path)

    print("\n" * 3 + "=" * 40)
    print(f"args: {args}")
    print(f"device: {device}")
    print(f"model: {model_id}")
    print(f"model size: {sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 1024 / 1024:.2f} MB")
    print(f"testset length: {len(testset)}")
    print("=" * 40 + "\n" * 3)

    outputfile = output_path / "det_metrics.csv"
    assert not outputfile.exists(), f"outputfile {outputfile} already exists"

    for sample in tqdm(testset):
        image: Image.Image = sample["image"]
        image_id: int = sample["image_id"]
        true_bboxes: dict = sample["objects"]
        true_bboxes = {k.replace("bbox", "boxes").replace("label", "labels"): v for k, v in true_bboxes.items()}  # rename to match model

        output = {}

        # preprocessing
        image = image.convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # runtime
        def inference():
            with torch.inference_mode():
                outputs = model(**inputs)
            return outputs

        timer = Timer(
            stmt="inference()",
            globals={"inference": inference, "model": model, "inputs": inputs, "torch": torch},
        )
        measurement = timer.blocked_autorange()
        output["inference_time"] = measurement.mean

        # postprocessing
        outputs = inference()
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=args.inference_threshold)[0]
        results = {k: v.cpu() for k, v in results.items()}
        results["labels_names"] = [model.config.id2label[elem] for elem in [elem.cpu().item() for elem in results["labels"]]]

        # evaluation
        pred_scores = results["scores"]  # [float]
        pred_labels = results["labels"]  # [int]
        pred_bboxes = results["boxes"]  # [x, y, w, h]
        true_labels = true_bboxes["labels"]  # [int]
        true_bboxes = true_bboxes["boxes"]  # [x, y, w, h]
        output["precision"] = compute_precision(pred_labels, pred_bboxes, true_labels, true_bboxes, args.precision_threshold)

        # save output
        with open(outputfile, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=output.keys())
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(output)


if __name__ == "__main__":
    args = SimpleNamespace(
        inference_threshold=0.5,
        precision_threshold=0.5,
    )
    main(args)
