---
title: "Assignment 1"
subtitle: "194.125 – AI/ML in the Era of Climate Change 2024W"
output: pdf_document
documentclass: article
papersize: a4
fontsize: 10pt
geometry:
    - top=10mm
    - bottom=15mm
    - left=10mm
    - right=10mm
toc: false
---

Table of Contents:

- Introduction: Brief about Object detection models (ODMs), LLMs, and their applications
- Background: The need for quantization, challenges in deploying ODMS and  LLMs, and an overview of quantization techniques
- Experiments: Explained setup and goals of the experiments
- Results: Detailed results from the quantization, including the benefits and any trade-offs. It should include at least 3 plots:
- Results 1.1:
    - Model Size (MB) vs Type of ODM (type and quantization)
    - Accuracy metric vs Type of ODM (type and quantization)
    - Inference time vs Type of ODM (type and quantization)
- Results 1.2:
    - Model Size (MB) vs Type of LLM (type and quantization)
    - Accuracy vs. Type of LLM (type and quantization)
    - Tokens/s vs. Type of LLM (type and quantization)
- Conclusions: Insights gained from the project, potential implications, and future recommendations

---

# Results 1.1

Tasks:

- selecting a pre-trained object detection model of our choice for the COCO-2017 dataset
- quantizing the model with LiteRT (formerly TensorFlow Lite) and the configurations [float32, float16, int8]
- measuring the accuracy and computational cost
    - Accuracy: average precision (see: https://cocodataset.org/#detection-eval) or pick up another one (...with a motivation!)
    - Inference time: seconds
    - Memory: model size (MB)
- writing the report & presentation

---

COCO dataset 2017 leaderboard:

- https://paperswithcode.com/sota/object-detection-on-coco
- https://github.com/ETH-DISCO/advx-bench/tree/main/analysis/model-selection

DETR family performs the best. "facebook/detr-resnet-101-dc5" in particular seems to also generalize across multiple datasets and be both zero-shot and open vocabulary. (But theoretically we could take any model of our choice).

# Results 1.2

Tasks:

- selecting a pre-trained large language model of our choice for the LAMBADA dataset
- quantizing the model with AutoGPTQ and the configurations [int8, int4, int2]
- measuring the accuracy and computational cost
    - Accuracy: Top-k Accuracy
    - Speed: Tokens/s
    - Memory: model size (MB)
- writing the report & presentation

---

LAMBADA leaderboard:

- https://paperswithcode.com/sota/language-modelling-on-lambada

