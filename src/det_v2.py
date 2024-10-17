import tensorflow_datasets as tfds
import numpy as np
import os
import time
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset
from utils import set_env

set_env(seed=42)
output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


def main(args: dict):
    #
    # dataset
    #

    dataset, info = tfds.load('coco/2017', with_info=True, data_dir=dataset_path)
    testset = dataset['test']

    def preprocess_image(example):
        image = example['image']
        image = tf.image.resize(image, [512, 512])
        image = tf.cast(image, tf.float32) / 255.0
        
        objects = example['objects']
        labels = objects['label']
        bboxes = objects['bbox']
        
        return image, {'labels': labels, 'bboxes': bboxes}

    testset = testset.map(preprocess_image)
    testset = testset.take(10) # set to 1500
    print(f"testset size: {len(list(testset))}")

    #
    # models
    #

    model = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d0/1")
    model_path = weights_path / "efficientdet"
    if not model_path.exists():
        tf.saved_model.save(model, str(model_path))
    print(f"original model size: {sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file()) / 1024 / 1024:.2f} MB")

    configs = ["float32", "float16", "int8"]
    for config in configs:
        tflite_model_path = weights_path / f"efficientdet_{config}.tflite"
        
        if not tflite_model_path.exists():
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
            
            if config == "float32":
                pass
            elif config == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif config == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                def representative_dataset(dataset):
                    num_samples = 10
                    for image, _ in tqdm(dataset.batch(1).take(num_samples), total=num_samples, desc="int8 sampling progress"):
                        yield [np.uint8(image.numpy() * 255)]
                converter.representative_dataset = lambda: representative_dataset(testset)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)

        print(f"{config} model size: {tflite_model_path.stat().st_size / 1024 / 1024:.2f} MB")

    #
    # eval loop
    #

    def run_inference(interpreter, image, config):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Prepare input data
        input_data = np.expand_dims(image.numpy(), axis=0)
        if config == "int8":
            input_data = np.uint8(input_data * 255)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get output tensors
        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

        # Post-process outputs
        boxes = boxes[0][:num_detections]
        classes = classes[0][:num_detections]
        scores = scores[0][:num_detections]

        return boxes, classes, scores, num_detections


    config = "float32"
    print(f"evaluating {config} model")
    tflite_model_path = weights_path / f"efficientdet_{config}.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    for image, ground_truth in tqdm(testset):
        # print(image.shape) # (512, 512, 3)
        # print(ground_truth) # {'labels': <tf.Tensor: shape=(0,), dtype=int64, numpy=array([], dtype=int64)>, 'bboxes': <tf.Tensor: shape=(0, 4), dtype=float32, numpy=array([], shape=(0, 4), dtype=float32)>}

        boxes, classes, scores, num_detections = run_inference(interpreter, image, config)















def main2(args: dict):
    #
    # dataset
    #

    dataset, info = tfds.load('coco/2017', with_info=True, data_dir=dataset_path)
    testset = dataset['test']

    def preprocess_image(example):
        image = example['image']
        image = tf.image.resize(image, [512, 512])
        image = tf.cast(image, tf.float32) / 255.0
        
        objects = example['objects']
        labels = objects['label']
        bboxes = objects['bbox']
        
        return image, {'labels': labels, 'bboxes': bboxes}

    testset = testset.map(preprocess_image)
    testset = testset.take(10) # set to 1500
    print(f"testset size: {len(list(testset))}")

    #
    # models
    #

    model = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d0/1")
    model_path = weights_path / "efficientdet"
    if not model_path.exists():
        tf.saved_model.save(model, str(model_path))
    print(f"original model size: {sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file()) / 1024 / 1024:.2f} MB")

    configs = ["float32", "float16", "int8"]
    for config in configs:
        tflite_model_path = weights_path / f"efficientdet_{config}.tflite"
        
        if not tflite_model_path.exists():
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
            
            if config == "float32":
                pass
            elif config == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif config == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                def representative_dataset(dataset):
                    num_samples = 10
                    for image, _ in tqdm(dataset.batch(1).take(num_samples), total=num_samples, desc="int8 sampling progress"):
                        yield [np.uint8(image.numpy() * 255)]
                converter.representative_dataset = lambda: representative_dataset(testset)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)

        print(f"{config} model size: {tflite_model_path.stat().st_size / 1024 / 1024:.2f} MB")

    #
    # eval loop
    #

    def run_inference(interpreter, image, config):
        # TODO: implement this
        return boxes, classes, scores, num_detections

    configs = ["float32", "float16", "int8"]
    for config in configs:
        print(f"evaluating {config} model")
        tflite_model_path = weights_path / f"efficientdet_{config}.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()

        for image, ground_truth in tqdm(testset):
            # print(image.shape) # (512, 512, 3)
            # print(ground_truth) # {'labels': <tf.Tensor: shape=(0,), dtype=int64, numpy=array([], dtype=int64)>, 'bboxes': <tf.Tensor: shape=(0, 4), dtype=float32, numpy=array([], shape=(0, 4), dtype=float32)>}

            boxes, classes, scores, num_detections = run_inference(interpreter, image, config)


if __name__ == "__main__":
    args = SimpleNamespace(
        inference_threshold=0.5,
        precision_threshold=0.5,
        # usually +100 samples for int8 calibration
    )
    main(args)
