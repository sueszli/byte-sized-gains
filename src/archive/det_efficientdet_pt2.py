"""
implementation of int8 quantization

- after a 2 full days of debugging it turned out that this very specific model breaks the quantizer and only returns zeros.
- i decided to start from scratch with a smaller model.
- docs: https://ai.google.dev/edge/litert/models/post_training_integer_quant
"""

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
# data
#

coco_dataset, coco_info = tfds.load("coco/2017", with_info=True, data_dir=dataset_path)


def preprocess_image(data):  # don't change this
    image = tf.image.resize(data["image"], (300, 300))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.cast(image, tf.uint8)
    return image


#
# models
#

model_path = weights_path / "efficientdet"
if not model_path.exists():
    model = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d0/1")
    tf.saved_model.save(model, str(model_path))
model = tf.saved_model.load(str(model_path))

quant_model_path = weights_path / f"efficientdet_int8.tflite"
if not quant_model_path.exists():
    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    representative_dataset = coco_dataset["train"].map(preprocess_image).batch(1).take(10)

    def representative_data_gen():
        for data in tqdm(representative_dataset):
            yield [data]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_converter = True
    tflite_model_quant = converter.convert()
    with open(quant_model_path, "wb") as f:
        f.write(tflite_model_quant)

#
# eval
#

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
