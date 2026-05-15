from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnxruntime as ort


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "producer" / "dist" / "iris_classifier.onnx"


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Model file not found: {candidate}")
    return candidate


def predict_label(features: Iterable[float], model_path: str | Path | None = None) -> int:
    values = list(features)
    if len(values) != 4:
        raise ValueError("The iris model expects exactly four numeric features.")

    session = ort.InferenceSession(str(resolve_model_path(model_path)), providers=["CPUExecutionProvider"])
    label_output, _ = session.run(None, {"features": np.asarray([values], dtype=np.float32)})
    return int(np.asarray(label_output).reshape(-1)[0])
