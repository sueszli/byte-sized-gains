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
toc: true
---

# 1 Quantization of an Object Detection Model

First step is finding the best performing model on the COCO dataset 2017.

So we look up:

- https://paperswithcode.com/sota/object-detection-on-coco
- https://github.com/ETH-DISCO/advx-bench/tree/main/analysis/model-selection

Turns out the best performing ones are of the DETR family. "facebook/detr-resnet-101-dc5" seems to perform and generalize the best.
