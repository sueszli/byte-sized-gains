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
toc: true
---

---

<!--

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

-->

# Introduction & Background

In recent years, object detection models (ODMs) and large language models (LLMs) have revolutionized computer vision and natural language processing, respectively. These advanced neural networks have found applications across various domains, from autonomous vehicles and surveillance systems to chatbots and content generation. However, the increasing complexity and size of these models have led to significant computational and energy demands, posing challenges for widespread deployment and raising concerns about their environmental impact.

As the global community grapples with climate change and the urgent need for sustainable technologies, the optimization of ODMs and LLMs has become a critical area of research. Quantization, a technique that reduces the precision of model parameters and activations, has emerged as a promising approach to address these challenges. By converting high-precision floating-point representations to lower-precision formats, quantization can substantially decrease model size, memory usage, and computational requirements without significantly compromising performance.

Our project focuses on applying advanced quantization techniques to both ODMs and LLMs, with the dual objectives of **improving their efficiency and reducing their carbon footprint**. We aim to implement quantization methods that enable these models to run on resource-constrained devices while maintaining their accuracy and functionality. By optimizing these models for deployment on edge devices and less powerful hardware, we can reduce the need for energy-intensive cloud computing and data centers, contributing to a more eco-friendly computational landscape.

The quantization of ODMs presents unique challenges due to the spatial nature of visual data and the need for precise localization in detection tasks. For LLMs, the primary obstacles lie in preserving the nuanced relationships between words and maintaining coherence across long sequences. Our project these model-specific issues while exploring commonalities in quantization strategies that can be applied across both domains. Through this project, we seek to demonstrate that quantization can play a crucial role in making advanced machine learning models more accessible and environmentally sustainable. By reducing the computational and energy requirements of ODMs and LLMs, we aim to pave the way for their widespread adoption in eco-conscious applications, ultimately contributing to global efforts in mitigating climate change and promoting sustainable technological development.

# Experiments & Results

## Results 1.1

<!--

Tasks:

- selecting a pre-trained object detection model of our choice for the COCO-2017 dataset
- quantizing the model with LiteRT (formerly TensorFlow Lite) and the configurations [float32, float16, int8]
- measuring the accuracy and computational cost (of > 1000 images)
    - Accuracy: average precision (see: https://cocodataset.org/#detection-eval) or pick up another one (...with a motivation!)
    - Inference time: seconds
    - Memory: model size (MB)
- writing the report & presentation

-->

model sizes:

```
original model: 31.88 MB
int8 model: 7.21 MB
float16 model: 12.17 MB
float32 model: 23.72 MB
```

benchmark machine specifications:

```
Tensorflow Version: 2.17.0

OS: macOS 14.6.1 23G93 arm64
Host: Mac14,10
Kernel: 23.6.0
CPU: Apple M2 Pro
GPU: Apple M2 Pro
Memory: 16384MiB
```

We started this task off by aiming for the stars and comparing the best models we could find on the public "papers with code" leaderboard for the COCO 2017 dataset [^coco]. Then we ran our own experiments to find the most representative models from each architecture family [^family]. We then noticed the DETR family to perform the best, particularly the "facebook/detr-resnet-101-dc5" model, as it also generalizes across multiple datasets and is both zero-shot and open vocabulary. This specific DETR model additionally was trained on COCO 2017 dataset which should give it an advantage.

[^coco]: https://paperswithcode.com/sota/object-detection-on-coco
[^family]: https://github.com/ETH-DISCO/advx-bench/tree/main/analysis/model-selection

But after implementing the entire evaluation pipeline for our experiments in PyTorch we realized Torch XLA builds a shared library, `_XLAC.so` that needs to link to the version of Python it was built with (currently 3.10 or 3.11). And in order to ensure that `import _XLAC` can succeed, we had to update the `LD_LIBRARY_PATH` to the lib directory of our Python environment. This was a major blocker for us as we were unable to resolve this issue even within a docker container. This made us have to pivot to TensorFlow models instead as they are directly supported by LiteRT and do not need a seperate layer of abstraction and translation like PyTorch models do.

We started from scratch, but this time instead of looking for state-of-the-art performance we were solely looking for models compatible with the very specific LiteRT quantization tool. We found that the models that are supported by LiteRT are very limited and the only models that are supported are the ones that are available in the TensorFlow model zoo. We initially started off by using Efficientnet but stumbled upon 0-gradient bugs in the int8 quantified version as the model is quite deep. We spent 2 full days trying to mitigate these issues but ended up pivoting again, but this time to the SSD family of models. We found that the SSD family of models are well supported by LiteRT and are also quite lightweight and performant.

Finally we decided to use `mobilenet_v2` as our base model for the quantization experiments. We quantized the model with LiteRT and the configurations [float32, float16, int8].

All previous attempts for each exercise were documented in the repository.

## Results 1.2

<!--
Tasks:

- selecting a pre-trained large language model of our choice for the LAMBADA dataset
- quantizing the model with AutoGPTQ and the configurations [int8, int4, int2]
- measuring the accuracy and computational cost
    - Accuracy: Top-k Accuracy
    - Speed: Tokens/s
    - Memory: model size (MB)
- writing the report & presentation

-->

LAMBADA leaderboard:

- https://paperswithcode.com/sota/language-modelling-on-lambada
