---
title: "Assignment 1"
subtitle: "194.125 – AI/ML in the Era of Climate Change 2024W"
author: "Code: [`github.com/sueszli/byte-sized-gains/`](https://github.com/sueszli/byte-sized-gains/)"
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
- Background: The need for quantization, challenges in deploying ODMS and LLMs, and an overview of quantization techniques
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
- measuring the accuracy and computational cost (of > 1000 images)
    - Accuracy: average precision (see: https://cocodataset.org/#detection-eval) or pick up another one (...with a motivation!)
    - Inference time: seconds
    - Memory: model size (MB)
- writing the report & presentation

---

We started this task off by aiming for the stars and comparing the best models we could find on the public "papers with code" leaderboard for the COCO 2017 dataset [^coco]. Then we ran our own experiments to find the most representative models from each architecture family [^family]. We then noticed the DETR family to perform the best, particularly the "facebook/detr-resnet-101-dc5" model, as it also generalizes across multiple datasets and is both zero-shot and open vocabulary. This specific DETR model additionally was trained on COCO 2017 dataset which should give it an advantage.

[^coco]: https://paperswithcode.com/sota/object-detection-on-coco
[^family]: https://github.com/ETH-DISCO/advx-bench/tree/main/analysis/model-selection

But after implementing the entire evaluation pipeline for our experiments in PyTorch we realized Torch XLA builds a shared library, `_XLAC.so` that needs to link to the version of Python it was built with (currently 3.10 or 3.11). And in order to ensure that `import _XLAC` can succeed, we had to update the `LD_LIBRARY_PATH` to the lib directory of our Python environment. This was a major blocker for us as we were unable to resolve this issue even within a docker container. This made us have to pivot to TensorFlow models instead as they are directly supported by LiteRT and do not need a seperate layer of abstraction and translation like PyTorch models do.

We started from scratch, but this time instead of looking for state-of-the-art performance we were solely looking for models compatible with the very specific LiteRT quantization tool. We found that the models that are supported by LiteRT are very limited and the only models that are supported are the ones that are available in the TensorFlow model zoo. We initially started off by using Efficientnet but stumbled upon 0-gradient bugs in the int8 quantified version as the model is quite deep. We spent 2 full days trying to mitigate these issues but ended up pivoting again, but this time to the SSD family of models. We found that the SSD family of models are well supported by LiteRT and are also quite lightweight and performant.

Finally we decided to use `mobilenet_v2` as our base model for the quantization experiments. We quantized the model with LiteRT and the configurations [float32, float16, int8]. We measured the accuracy and computational cost of a few images to make sure the model is working as expected. This time we were fortuate enough to be able to conduct our experiments successfully after just a few hours. All previous attempts for each exercise were documented in the repository.

... write about experiments and results ...

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

