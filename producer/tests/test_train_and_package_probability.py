from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

from producer.train_and_package_probability import (
    METADATA_FILE_NAME,
    MODEL_FILE_NAME,
    train_and_package,
)


def test_train_and_package_probability_exports_a_working_model(tmp_path):
    model_path = train_and_package(tmp_path)

    assert model_path.name == MODEL_FILE_NAME
    assert model_path.exists()

    metadata = json.loads((tmp_path / METADATA_FILE_NAME).read_text(encoding="utf-8"))
    assert metadata["dataset"] == "iris"
    assert metadata["contract_version"] == 1
    assert metadata["input_name"] == "features"
    assert metadata["feature_count"] == 4
    assert metadata["output_name"] == "setosa_probability"
    assert metadata["output_kind"] == "probability"
    assert metadata["prediction_tolerance"] == 0.001

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (probability_output,) = session.run(
        None,
        {"features": np.asarray([metadata["sample_input"]], dtype=np.float32)},
    )

    prediction = float(np.asarray(probability_output).reshape(-1)[0])
    assert abs(prediction - metadata["expected_sample_prediction"]) <= metadata["prediction_tolerance"]
    assert 0.0 <= prediction <= 1.0
