from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort

from producer.train_and_package import METADATA_FILE_NAME, MODEL_FILE_NAME, train_and_package


def test_train_and_package_exports_a_working_model(tmp_path):
    model_path = train_and_package(tmp_path)

    assert model_path.name == MODEL_FILE_NAME
    assert model_path.exists()

    metadata = json.loads((tmp_path / METADATA_FILE_NAME).read_text(encoding="utf-8"))
    assert metadata["dataset"] == "iris"

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    label_output, _ = session.run(
        None,
        {"features": np.asarray([metadata["sample_input"]], dtype=np.float32)},
    )

    assert int(np.asarray(label_output).reshape(-1)[0]) == metadata["expected_sample_prediction"]
