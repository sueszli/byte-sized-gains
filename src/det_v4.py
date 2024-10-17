"""
finally got int8 quantization to work for mobilenetv2 - time to bring everything together
"""

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
set_env(seed=42)
output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


# mobilenetv2
model_path = weights_path / "mobilenetv2"
if not model_path.exists():
    model = kagglehub.model_download(handle="tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2", path=str(model_path))
    tf.saved_model.save(model, str(model_path))
model = tf.saved_model.load(str(model_path))

# coco2017
coco_dataset, coco_info = tfds.load("coco/2017", with_info=True, data_dir=dataset_path)


def preprocess_image(data):  # don't change this
    image = tf.image.resize(data["image"], (300, 300))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.cast(image, tf.uint8)
    return image


representative_dataset = coco_dataset["train"].map(preprocess_image).batch(1).take(10)

# quantize
converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))


def representative_data_gen():
    for data in tqdm(representative_dataset):
        yield [data]


quant_model_path = weights_path / "mobilenetv2_int8.tflite"
if not quant_model_path.exists():
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = True
    tflite_model_quant = converter.convert()

    with open(quant_model_path, "wb") as f:
        f.write(tflite_model_quant)

# evaluate
test_dataset = coco_dataset["test"].map(preprocess_image).batch(1).take(1)
test_image = next(iter(test_dataset))[0]
test_image = test_image.numpy()

interpreter = tf.lite.Interpreter(model_path=str(quant_model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()

input_shape = input_details["shape"]
test_image = tf.image.resize(test_image, (input_shape[1], input_shape[2]))

if input_details["dtype"] == np.uint8:
    input_scale, input_zero_point = input_details["quantization"]
    if input_scale != 0:
        test_image = (test_image - input_zero_point) * input_scale
    else:
        test_image = test_image - input_zero_point

test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
interpreter.set_tensor(input_details["index"], test_image)
interpreter.invoke()

outputs = [interpreter.get_tensor(output["index"]) for output in output_details]
print(outputs)
