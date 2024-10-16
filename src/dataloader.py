
from PIL import Image
from utils import get_device
from tqdm import tqdm
from utils import set_env
import os
import json
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torch
from pathlib import Path
from transformers import AutoImageProcessor, DetrForObjectDetection
from datasets import load_dataset

set_env(seed=42)

batch_size = 512
threshold = 0.8

classes_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(classes_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


# 
# data
# 

testset = load_dataset("phiyodr/coco2017", split="validation", streaming=False, cache_dir=dataset_path)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

# 
# model
# 

# model_id = "facebook/detr-resnet-101-dc5"  # largest model
# device = get_device(disable_mps=False)

# image_processor = AutoImageProcessor.from_pretrained(model_id, device=device)
# model = DetrForObjectDetection.from_pretrained(model_id, device=device, local_files_only=True, cache_dir=weights_path, threshold=threshold, return_labels_only=True)
# model.eval()


# inputs = image_processor(images=img, return_tensors="pt")
# inputs = {k: v.to(device) for k, v in inputs.items()}

# with torch.no_grad():
#     outputs = model(**inputs)

# target_sizes = torch.tensor([img.size[::-1]]).to(device)
# results = image_processor.post_process_object_detection(outputs, threshold, target_sizes=target_sizes)[0]

# results["boxes"] = [elem.cpu().tolist() for elem in results["boxes"]]
# results["scores"] = [elem.cpu().item() for elem in results["scores"]]
# results["labels"] = [model.config.id2label[elem.cpu().item()] for elem in results["labels"]]

# # model_labels = [model.config.id2label[elem.item()] for elem in results["labels"]]
# # results["labels"] = []
# # for ml in model_labels:
# #     for cl in labels:
# #         if cl.lower() in ml.lower():
# #             results["labels"].append(cl)
# #             break
# #     else:
# #         results["labels"].append("unknown")

# boxes = results["boxes"]
# scores = results["scores"]
# labels = results["labels"]

# # Assertions for type checking
# assert all(isinstance(label, str) for label in labels)
# assert all(isinstance(score, float) for score in scores)
# assert all(isinstance(box, list) and len(box) == 4 for box in boxes)
# assert all(all(isinstance(coord, float) for coord in box) for box in boxes)

