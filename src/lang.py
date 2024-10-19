import time
import csv
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

logging.getLogger("tensorflow").setLevel(logging.DEBUG)
output_path = Path.cwd() / "data"
dataset_path = Path.cwd() / "datasets"
weights_path = Path.cwd() / "weights"

os.makedirs(output_path, exist_ok=True)
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(weights_path, exist_ok=True)


def main(args):
    # quantize weights only
    configs = ["int8", "int4", "int2"]


if __name__ == "__main__":
    args = SimpleNamespace(
        sample_size=1500,
    )
    main(args)
