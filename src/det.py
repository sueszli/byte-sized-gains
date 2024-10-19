import csv
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import kagglehub
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

logging.getLogger("tensorflow").setLevel(logging.DEBUG)
output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


def normalize_boxes(boxes, input_shape):
    height, width = input_shape[1], input_shape[2]
    normalized_boxes = []

    for box in boxes:
        ymin, xmin, ymax, xmax = box

        # normalize coordinates
        xmin_norm = max(0, min(1, xmin / width))
        ymin_norm = max(0, min(1, ymin / height))
        xmax_norm = max(0, min(1, xmax / width))
        ymax_norm = max(0, min(1, ymax / height))

        # reorder to [xmin, ymin, xmax, ymax]
        normalized_box = [xmin_norm, ymin_norm, xmax_norm, ymax_norm]
        normalized_boxes.append(normalized_box)

    return normalized_boxes


def compute_precision(pred_labels, pred_bboxes, true_labels, true_bboxes, iou_threshold):
    true_positives = 0
    false_positives = 0

    def compute_iou(box1, box2):
        y1_max = max(box1[0], box2[0])
        x1_max = max(box1[1], box2[1])
        y2_min = min(box1[2], box2[2])
        x2_min = min(box1[3], box2[3])

        intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection / (area1 + area2 - intersection + 1e-6)
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


def print_outputdetails(outputs, output_details):  # debug
    for i, output in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {output['name']}")
        print(f"  Shape: {output['shape']}")
        print(f"  Values: {outputs[i]}")
    print("-" * 80)


def main(args):
    #
    # data
    #

    def preprocess_image(data):  # don't change this
        image = tf.image.resize(data["image"], (300, 300))
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.cast(image, tf.uint8)
        return image

    coco_dataset, coco_info = tfds.load("coco/2017", with_info=True, data_dir=dataset_path)

    #
    # models
    #

    # original model
    model_path = weights_path / "mobilenetv2"
    if not model_path.exists():
        model = kagglehub.model_download(handle="tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2", path=str(model_path))
        tf.saved_model.save(model, str(model_path))
    model = tf.saved_model.load(str(model_path))
    print(f"original model: {sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file()) / 1024 / 1024:.2f} MB")

    # quantized models
    configs = ["int8", "float16", "float32"]
    for config in configs:
        quant_model_path = weights_path / f"mobilenetv2_{config}.tflite"

        if not quant_model_path.exists():
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))

            if config == "float32":
                pass
            elif config == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif config == "int8":

                def representative_data_gen():
                    for data in tqdm(representative_dataset):
                        yield [data]

                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                representative_dataset = coco_dataset["train"].map(preprocess_image).batch(1).take(args.int8_train_size)
                converter.representative_dataset = representative_data_gen
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                converter.experimental_new_converter = True
            tflite_model_quant = converter.convert()

            with open(quant_model_path, "wb") as f:
                f.write(tflite_model_quant)

        print(f"{config} model: {quant_model_path.stat().st_size / 1024 / 1024:.2f} MB")

    #
    # eval
    #

    configs = ["int8", "float16", "float32"]
    for config in configs:
        print(f"evaluating {config} model")

        quant_model_path = weights_path / f"mobilenetv2_{config}.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(quant_model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()
        input_shape = input_details["shape"]

        assert args.sample_size <= len(coco_dataset["validation"])  # test set doesn't have bboxes
        test_dataset = coco_dataset["validation"].batch(1).take(args.sample_size)
        for data in tqdm(test_dataset):
            test_image = preprocess_image(data)
            test_image = test_image[0]
            test_image = test_image.numpy()
            test_image = tf.image.resize(test_image, (input_shape[1], input_shape[2]))
            if input_details["dtype"] == np.uint8:
                input_scale, input_zero_point = input_details["quantization"]
                if input_scale != 0:
                    test_image = (test_image - input_zero_point) * input_scale
                else:
                    test_image = test_image - input_zero_point
            test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])

            # benchmark (don't use tf.timestamp)
            interpreter.set_tensor(input_details["index"], test_image)
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
            inference_time = end_time - start_time

            # dequantize
            outputs = [interpreter.get_tensor(output["index"]) for output in output_details]
            if config == "int8":
                for i, output in enumerate(output_details):
                    scale, zero_point = output["quantization"]
                    if scale != 0:
                        outputs[i] = (outputs[i].astype(np.float32) - zero_point) * scale

            # limit to num_detections
            num_detections = int(outputs[2][0])
            boxes = outputs[0][0][:num_detections].tolist()  # [ymin, xmin, ymax, xmax]
            classes = list(map(int, outputs[5][0][:num_detections].tolist()))
            scores = outputs[6][0][:num_detections].tolist()

            # filter by confidence threshold
            boxes = [box for box, score in zip(boxes, scores) if score > args.confidence_threshold]
            classes = [cls for cls, score in zip(classes, scores) if score > args.confidence_threshold]
            scores = [score for score in scores if score > args.confidence_threshold]

            # normalize
            classes = [x - 1 for x in classes]
            boxes = normalize_boxes(boxes, input_shape)

            true_boxes = data["objects"]["bbox"][0].numpy().tolist()
            true_classes = data["objects"]["label"][0].numpy().tolist()

            precision = compute_precision(pred_bboxes=boxes, pred_labels=classes, true_bboxes=true_boxes, true_labels=true_classes, iou_threshold=args.iou_threshold)

            result = {
                "quantization": config,
                "inference_time": inference_time,
                "precision": precision,
            }
            with open(output_path / "det.csv", "a") as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(result)


if __name__ == "__main__":
    args = SimpleNamespace(
        int8_train_size=100,
        sample_size=1500,
        confidence_threshold=0.0,
        iou_threshold=0.01,  # extremely sensitive
    )
    main(args)
