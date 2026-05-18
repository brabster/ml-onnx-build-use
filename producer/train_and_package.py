from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import onnx
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


MODEL_FILE_NAME = "regression_model.onnx"
PICKLE_MODEL_FILE_NAME = "regression_model.pkl"
METADATA_FILE_NAME = "model_metadata.json"

N_FEATURES = 10


def train_and_package(output_dir: Path) -> Path:
    features, targets = make_regression(
        n_samples=100_000,
        n_features=N_FEATURES,
        noise=10.0,
        random_state=42,
    )
    features = features.astype(np.float32)

    model = Ridge(random_state=42)
    model.fit(features, targets)

    onnx_model = to_onnx(
        model,
        initial_types=[("features", FloatTensorType([None, N_FEATURES]))],
        target_opset=18,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILE_NAME
    pickle_model_path = output_dir / PICKLE_MODEL_FILE_NAME
    metadata_path = output_dir / METADATA_FILE_NAME

    onnx.save_model(onnx_model, model_path)
    with pickle_model_path.open("wb") as model_file:
        pickle.dump(model, model_file)

    sample_input = features[0].tolist()
    metadata = {
        "dataset": "synthetic_regression",
        "n_features": N_FEATURES,
        "feature_names": [f"feature_{i}" for i in range(N_FEATURES)],
        "sample_input": sample_input,
        "expected_sample_prediction": float(model.predict([sample_input])[0]),
        "prediction_tolerance": 1.0,
        "model_file": MODEL_FILE_NAME,
        "pickle_model_file": PICKLE_MODEL_FILE_NAME,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the regression model and export it as ONNX.")
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
