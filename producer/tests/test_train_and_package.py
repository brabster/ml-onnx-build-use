from __future__ import annotations

import json
import pickle

import numpy as np
import onnxruntime as ort

from producer.train_and_package import (
    METADATA_FILE_NAME,
    MODEL_FILE_NAME,
    PICKLE_MODEL_FILE_NAME,
    train_and_package,
)


def test_train_and_package_exports_a_working_model(tmp_path):
    model_path = train_and_package(tmp_path)

    assert model_path.name == MODEL_FILE_NAME
    assert model_path.exists()
    pickle_model_path = tmp_path / PICKLE_MODEL_FILE_NAME
    assert pickle_model_path.exists()

    metadata = json.loads((tmp_path / METADATA_FILE_NAME).read_text(encoding="utf-8"))
    assert metadata["dataset"] == "synthetic_regression"
    assert metadata["pickle_model_file"] == PICKLE_MODEL_FILE_NAME

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    (variable_output,) = session.run(
        None,
        {"features": np.asarray([metadata["sample_input"]], dtype=np.float32)},
    )

    actual = float(np.asarray(variable_output).reshape(-1)[0])
    print(f"predict_value({metadata['sample_input']}) -> {actual:.4f} (expected {metadata['expected_sample_prediction']:.4f})")
    assert abs(actual - metadata["expected_sample_prediction"]) < metadata["prediction_tolerance"]

    with pickle_model_path.open("rb") as model_file:
        pickle_model = pickle.load(model_file)
    pickle_prediction = float(pickle_model.predict(np.asarray([metadata["sample_input"]], dtype=np.float32))[0])
    print(
        f"pickle_predict_value({metadata['sample_input']}) -> {pickle_prediction:.4f} "
        f"(expected {metadata['expected_sample_prediction']:.4f})"
    )
    assert abs(pickle_prediction - metadata["expected_sample_prediction"]) < metadata["prediction_tolerance"]
