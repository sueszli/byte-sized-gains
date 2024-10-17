# https://ai.google.dev/edge/litert/models/post_training_integer_quant

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm

from utils import set_env
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
print("tensorFlow version: ", tf.__version__)

set_env(seed=42)
output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


exit()


# save coco 2017
dataset, info = tfds.load("coco/2017", with_info=True, data_dir=dataset_path)
testset = dataset["test"].take(1500)
trainset = dataset["train"].take(1500)

train_labels = tf.data.Dataset.from_tensor_slices(list(map(lambda x: x['objects']['label'], trainset))).batch(1)
train_images = tf.data.Dataset.from_tensor_slices(list(map(lambda x: x['image'], trainset))).batch(1)
trainset = tf.data.Dataset.zip((train_images, train_labels))





# save efficientdet model
model = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d0/1")
model_path = weights_path / "efficientdet"
if not model_path.exists():
    tf.saved_model.save(model, str(model_path))

tflite_model_path = weights_path / f"efficientdet_int8.tflite"
if not tflite_model_path.exists():

    def representative_data_gen():
        for input_value in tqdm(tf.data.Dataset.from_tensor_slices(trainset).batch(1).take(10)):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model_quant)

# inference
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)

input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)


exit()

# def preprocess(image): # requirements of efficientdet (don't change)
#     image = tf.image.resize(image, (512, 512))
#     image = tf.cast(image, tf.float32) / 255.0
#     image = tf.cast(image, tf.uint8)
#     image = tf.expand_dims(image, axis=0)
#     return image

# int8 quantize efficientdet model
config = "int8"


if not tflite_model_path.exists():
    def representative_dataset_gen():
        for data in tqdm(testset.take(10)):
            yield [preprocess(data['image'])]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    tflite_model_path = weights_path / f"efficientdet_{config}.tflite"
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

# inference
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
print(f"New input_details: {input_details}")

exit()

input_type = interpreter.get_input_details()[0]['dtype']
output_type = interpreter.get_output_details()[0]['dtype']
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"input_type: {input_type}")
print(f"output_type: {output_type}")
print(f"input_details: {input_details}")
print(f"output_details: {output_details}")

for sample in tqdm(testset):
    test_image = sample['image']
    test_image = tf.image.resize(test_image, (512, 512))
    test_image = tf.cast(test_image, tf.float32) / 255.0
    test_image = tf.cast(test_image, tf.uint8)
    test_image = tf.expand_dims(test_image, axis=0)
    print(f"test_image after preprocessing: {test_image.shape}")

    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        if input_scale != 0:
            test_image = tf.cast(test_image, tf.float32)
            test_image = test_image / input_scale + input_zero_point
            test_image = tf.cast(test_image, tf.uint8)
            print(f"test_image after rescale: {test_image.shape}")
        else:
            test_image = tf.cast(test_image, tf.uint8) + input_zero_point
            print(f"test_image after rescale (is null): {test_image.shape}")

    test_image = test_image.numpy().astype(input_details["dtype"])
    print(f"test_image after np.expand_dims: {test_image.shape}")
    
    expected_shape = tuple(input_details['shape'])
    print(f"expected_shape: {expected_shape}")
    if test_image.shape != expected_shape:
        test_image = np.reshape(test_image, expected_shape)
        print(f"test_image after np.reshape: {test_image.shape}")

    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]









# # inference
# interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
# interpreter.allocate_tensors()

# input_type = interpreter.get_input_details()[0]['dtype']
# output_type = interpreter.get_output_details()[0]['dtype']
# input_details = interpreter.get_input_details()[0]
# output_details = interpreter.get_output_details()[0]

# print(f"input_type: {input_type}")
# print(f"output_type: {output_type}")
# print(f"input_details: {input_details}")
# print(f"output_details: {output_details}")

# for sample in tqdm(testset):
#     # same as: test_image = preprocess(sample['image'])
#     test_image = sample['image']
#     print(f"test_image: {test_image.shape}")
#     test_image = tf.image.resize(test_image, (512, 512))
#     print(f"test_image after tf.image.resize: {test_image.shape}")
#     test_image = tf.cast(test_image, tf.float32) / 255.0
#     print(f"test_image after tf.cast: {test_image.shape}")
#     test_image = tf.cast(test_image, tf.uint8)
#     print(f"test_image after tf.cast: {test_image.shape}")
#     test_image = tf.expand_dims(test_image, axis=0)
#     print(f"test_image after tf.expand_dims: {test_image.shape}")

#     # Check if the input type is quantized, then rescale input data to uint8
#     if input_details['dtype'] == np.uint8:
#         input_scale, input_zero_point = input_details["quantization"]
#         print(f"input_scale: {input_scale}")
#         print(f"input_zero_point: {input_zero_point}")
#         test_image = test_image / input_scale + input_zero_point
#         print(f"test_image after rescale: {test_image.shape}")

#     test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
#     print(f"test_image after np.expand_dims: {test_image.shape}")
#     interpreter.set_tensor(input_details["index"], test_image)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details["index"])[0]
