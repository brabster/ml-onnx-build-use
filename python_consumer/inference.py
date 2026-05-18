from __future__ import annotations

import os
import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnxruntime as ort


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "producer" / "dist" / "iris_classifier.onnx"
DEFAULT_PROBABILITY_MODEL_PATH = Path(__file__).resolve().parents[1] / "producer" / "dist" / "setosa_probability.onnx"
DEFAULT_PICKLE_MODEL_PATH = Path(__file__).resolve().parents[1] / "producer" / "dist" / "iris_classifier.pkl"


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Model file not found: {candidate}")
    return candidate


def resolve_probability_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path or os.environ.get("SETOSA_MODEL_PATH", DEFAULT_PROBABILITY_MODEL_PATH)).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Model file not found: {candidate}")
    return candidate


def resolve_pickle_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path or os.environ.get("PICKLE_MODEL_PATH", DEFAULT_PICKLE_MODEL_PATH)).expanduser()
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


def predict_label_from_pickle(features: Iterable[float], model_path: str | Path | None = None) -> int:
    values = list(features)
    if len(values) != 4:
        raise ValueError("The iris model expects exactly four numeric features.")

    with resolve_pickle_model_path(model_path).open("rb") as model_file:
        model = pickle.load(model_file)
    return int(model.predict(np.asarray([values], dtype=np.float32))[0])


def predict_setosa_probability(features: Iterable[float], model_path: str | Path | None = None) -> float:
    values = list(features)
    if len(values) != 4:
        raise ValueError("The iris model expects exactly four numeric features.")

    session = ort.InferenceSession(str(resolve_probability_model_path(model_path)), providers=["CPUExecutionProvider"])
    (probability_output,) = session.run(None, {"features": np.asarray([values], dtype=np.float32)})
    return float(np.asarray(probability_output).reshape(-1)[0])
