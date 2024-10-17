import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm

from utils import set_env

set_env(seed=42)
output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


#
# coco dataset
#

dataset, info = tfds.load("coco/2017", with_info=True, data_dir=dataset_path)
testset = dataset["test"]


def preprocess_image(example):
    image = example["image"]
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0
    objects = example["objects"]
    labels = objects["label"]
    bboxes = objects["bbox"]
    return image, {"labels": labels, "bboxes": bboxes}


testset = testset.map(preprocess_image)
testset = testset.take(10)

#
# int8 quantized efficientdet
#

model = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d0/1")
model_path = weights_path / "efficientdet"
if not model_path.exists():
    tf.saved_model.save(model, str(model_path))

config = "int8"
tflite_model_path = weights_path / f"efficientdet_{config}.tflite"

if not tflite_model_path.exists():
    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
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
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

#
# inference
#

config = "int8"
print(f"evaluating {config} model")
tflite_model_path = weights_path / f"efficientdet_{config}.tflite"
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

for image, ground_truth in tqdm(testset):
    image = (image * 255).numpy().astype(np.uint8)
    input_data = tf.expand_dims(image, axis=0)

    input_shape = input_details[0]["shape"]
    input_data = tf.image.resize(input_data, (input_shape[1], input_shape[2]))
    input_data = tf.cast(input_data, dtype=input_details[0]["dtype"])

    input_scale, input_zero_point = input_details[0]['quantization']
    input_data = (input_data / input_scale) + input_zero_point
    input_data = tf.cast(input_data, tf.uint8)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    
    # dequantize the output
    output_data = interpreter.get_tensor(output_details[0]["index"])
    output_scale, output_zero_point = output_details[0]['quantization']
    output_data = tf.cast(output_data, tf.float32)
    output_zero_point = tf.cast(output_zero_point, tf.float32)
    output_data = (output_data - output_zero_point) * output_scale
