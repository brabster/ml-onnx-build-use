from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

from producer.train_and_package_regression import (
    METADATA_FILE_NAME,
    MODEL_FILE_NAME,
    train_and_package,
)


def test_train_and_package_regression_exports_a_working_model(tmp_path):
    model_path = train_and_package(tmp_path)

    assert model_path.name == MODEL_FILE_NAME
    assert model_path.exists()

    metadata = json.loads((tmp_path / METADATA_FILE_NAME).read_text(encoding="utf-8"))
    assert metadata["dataset"] == "synthetic_make_regression"
    assert metadata["n_samples"] >= 100_000

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (prediction_output,) = session.run(
        None,
        {"features": np.asarray([metadata["sample_input"]], dtype=np.float32)},
    )

    prediction = float(np.asarray(prediction_output).reshape(-1)[0])
    print(
        f"predict_regression({metadata['sample_input']}) -> {prediction:.4f} "
        f"(expected {metadata['expected_sample_prediction']:.4f})"
    )
    assert abs(prediction - metadata["expected_sample_prediction"]) < 0.01
