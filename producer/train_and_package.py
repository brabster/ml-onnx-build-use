from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import onnx
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


MODEL_FILE_NAME = "iris_classifier.onnx"
PICKLE_MODEL_FILE_NAME = "iris_classifier.pkl"
METADATA_FILE_NAME = "model_metadata.json"


def train_and_package(output_dir: Path) -> Path:
    iris = load_iris()
    features = iris.data.astype(np.float32)
    labels = iris.target

    classifier = LogisticRegression(max_iter=400, random_state=0)
    classifier.fit(features, labels)

    onnx_model = to_onnx(
        classifier,
        initial_types=[("features", FloatTensorType([None, features.shape[1]]))],
        options={id(classifier): {"zipmap": False}},
        target_opset=18,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILE_NAME
    pickle_model_path = output_dir / PICKLE_MODEL_FILE_NAME
    metadata_path = output_dir / METADATA_FILE_NAME

    onnx.save_model(onnx_model, model_path)
    with pickle_model_path.open("wb") as model_file:
        pickle.dump(classifier, model_file)

    metadata = {
        "dataset": "iris",
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
        "sample_input": features[0].tolist(),
        "expected_sample_prediction": int(labels[0]),
        "model_file": MODEL_FILE_NAME,
        "pickle_model_file": PICKLE_MODEL_FILE_NAME,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the iris model and export it as ONNX.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dist",
        help="Directory that will receive the packaged ONNX model.",
    )
    arguments = parser.parse_args()
    model_path = train_and_package(arguments.output_dir)
    print(f"Packaged model written to {model_path}")


if __name__ == "__main__":
    main()
