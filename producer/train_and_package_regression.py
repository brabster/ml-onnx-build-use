from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


MODEL_FILE_NAME = "synthetic_regression.onnx"
METADATA_FILE_NAME = "synthetic_regression_metadata.json"


def train_and_package(output_dir: Path) -> Path:
    features, targets = make_regression(
        n_samples=100_000,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=42,
    )
    features = features.astype(np.float32)
    targets = targets.astype(np.float32)

    regressor = LinearRegression()
    regressor.fit(features, targets)

    onnx_model = to_onnx(
        regressor,
        initial_types=[("features", FloatTensorType([None, features.shape[1]]))],
        target_opset=18,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILE_NAME
    metadata_path = output_dir / METADATA_FILE_NAME

    onnx.save_model(onnx_model, model_path)

    sample_input = features[0].tolist()
    metadata = {
        "dataset": "synthetic_make_regression",
        "description": "Linear regression trained on scikit-learn make_regression synthetic data",
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "random_state": 42,
        "sample_input": sample_input,
        "expected_sample_prediction": float(regressor.predict(np.asarray([sample_input], dtype=np.float32))[0]),
        "model_file": MODEL_FILE_NAME,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a representative synthetic regression model and export it as ONNX."
    )
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
