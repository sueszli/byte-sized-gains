import os
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

    testset = load_dataset("rafaelpadilla/coco2017", split="val", streaming=False, cache_dir=dataset_path).take(1500)
    print(f"testset size: {len(testset)}")

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
        if tflite_model_path.exists():
            print(f"{config} tflite model size: {tflite_model_path.stat().st_size / 1024 / 1024:.2f} MB - already exists")
            continue

        converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))

        if config == "float32":
            pass  # default

        elif config == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif config == "int8":
            size = 50
            representative_dataset = []
            for sample in tqdm(list(testset.take(size))):
                image = sample["image"]
                if not isinstance(image, tf.Tensor):
                    image = tf.convert_to_tensor(image)
                if len(image.shape) == 2:
                    image = tf.expand_dims(image, -1)
                if len(image.shape) == 3:
                    image = tf.expand_dims(image, 0)
                image = tf.image.resize(image, (512, 512))
                image = tf.cast(image, tf.uint8)
                if image.shape[-1] == 1:
                    image = tf.image.grayscale_to_rgb(image)
                representative_dataset.append(image)

            def representative_dataset_gen():
                for i, img in enumerate(representative_dataset):
                    print(f"int8 progress: {i}/{size}")
                    yield [img]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print(f"{config} tflite model size: {tflite_model_path.stat().st_size / 1024 / 1024:.2f} MB")

    exit()

    #
    # eval loop
    #

    for config in configs:
        # load model
        model = tf.lite.Interpreter(model_path=weights_path / f"efficientdet_{config}.tflite")

        for sample in tqdm(testset):
            image: Image.Image = sample["image"]
            image_id: int = sample["image_id"]
            true_bboxes: dict = sample["objects"]  # {"bbox": [x, y, w, h], "label": int}


if __name__ == "__main__":
    args = SimpleNamespace(
        inference_threshold=0.5,
        precision_threshold=0.5,
    )
    main(args)
