from __future__ import annotations

import os
import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnxruntime as ort


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "producer" / "dist" / "regression_model.onnx"
DEFAULT_PICKLE_MODEL_PATH = Path(__file__).resolve().parents[1] / "producer" / "dist" / "regression_model.pkl"

N_FEATURES = 10


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Model file not found: {candidate}")
    return candidate


def resolve_pickle_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path or os.environ.get("PICKLE_MODEL_PATH", DEFAULT_PICKLE_MODEL_PATH)).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Model file not found: {candidate}")
    return candidate


def predict_value(features: Iterable[float], model_path: str | Path | None = None) -> float:
    values = list(features)
    if len(values) != N_FEATURES:
        raise ValueError(f"The regression model expects exactly {N_FEATURES} numeric features.")

    session = ort.InferenceSession(str(resolve_model_path(model_path)), providers=["CPUExecutionProvider"])
    (variable_output,) = session.run(None, {"features": np.asarray([values], dtype=np.float32)})
    return float(np.asarray(variable_output).reshape(-1)[0])


def predict_value_from_pickle(features: Iterable[float], model_path: str | Path | None = None) -> float:
    values = list(features)
    if len(values) != N_FEATURES:
        raise ValueError(f"The regression model expects exactly {N_FEATURES} numeric features.")

    with resolve_pickle_model_path(model_path).open("rb") as model_file:
        model = pickle.load(model_file)
    return float(model.predict(np.asarray([values], dtype=np.float32))[0])
