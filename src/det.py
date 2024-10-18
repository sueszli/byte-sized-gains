"""
for some reason the precision is always zero
"""


from types import SimpleNamespace
import logging
import os
from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from utils import set_env

logging.getLogger("tensorflow").setLevel(logging.DEBUG)
# set_env(seed=42)
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
        # Convert [ymin, xmin, ymax, xmax] to [x1, y1, x2, y2]
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[1], box1[0], box1[3], box1[2]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[1], box2[0], box2[3], box2[2]

        # Calculate intersection area
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area

        # Calculate IoU
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

        # test set doesn't have bboxes
        assert args.sample_size <= len(coco_dataset["validation"])
        test_dataset = coco_dataset["validation"].batch(1).take(args.sample_size)
        for data in tqdm(test_dataset):
            # benchmark
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
            start_time = tf.timestamp()
            interpreter.set_tensor(input_details["index"], test_image)
            interpreter.invoke()
            end_time = tf.timestamp()
            inference_time = end_time - start_time

            # inference
            true_bboxes = data["objects"]["bbox"]
            true_classes = data["objects"]["label"]

            outputs = [interpreter.get_tensor(output["index"]) for output in output_details]
            if config == "int8": # dequantize
                for i, output in enumerate(output_details):
                    scale, zero_point = output['quantization']
                    if scale != 0:
                        outputs[i] = (outputs[i].astype(np.float32) - zero_point) * scale
            pred_bboxes = outputs[0] # [ymin, xmin, ymax, xmax] in range 
            pred_scores = outputs[4]
            pred_classes = outputs[5]
        
            # filter by confidence threshold
            confidence_threshold = args.confidence_threshold
            max_scores = np.max(pred_scores, axis=-1)  # Shape: (1, 100)
            mask = max_scores > confidence_threshold  # Shape: (1, 100)
            pred_classes = pred_classes[mask]
            pred_scores = max_scores[mask]
            pred_bboxes = pred_bboxes[:, :100][mask] # assume first 100 to align
            pred_bboxes = pred_bboxes.reshape(1, -1, 4)
            pred_classes = pred_classes.reshape(1, -1)
            pred_scores = pred_scores.reshape(1, -1)

            # compute precision
            precision = compute_precision(pred_classes[0], pred_bboxes[0], true_classes.numpy()[0], true_bboxes.numpy()[0], args.iou_threshold)
            print(f"precision: {precision}, inference time: {inference_time:.2f}s")


def print_outputdetails(outputs, output_details):
    for i, output in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {output['name']}")
        print(f"  Shape: {output['shape']}")
        print(f"  Values: {outputs[i]}")
    print("-" * 80)


if __name__ == "__main__":
    args = SimpleNamespace(
        int8_train_size = 100,
        sample_size = 10, # set to 1500 for full evaluation
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )
    main(args)
