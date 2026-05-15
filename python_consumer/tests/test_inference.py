import json

from python_consumer.inference import (
    predict_label,
    predict_setosa_probability,
    resolve_model_path,
    resolve_probability_model_path,
)

# Metadata filenames are the contract: both producer and consumer must use the same names.
_IRIS_METADATA_FILE = "model_metadata.json"
_SETOSA_METADATA_FILE = "setosa_probability_metadata.json"


def _load_iris_metadata() -> dict:
    model_path = resolve_model_path()
    return json.loads((model_path.parent / _IRIS_METADATA_FILE).read_text(encoding="utf-8"))


def _load_setosa_metadata() -> dict:
    model_path = resolve_probability_model_path()
    return json.loads((model_path.parent / _SETOSA_METADATA_FILE).read_text(encoding="utf-8"))


def test_python_consumer_predicts_label_using_contract_sample():
    metadata = _load_iris_metadata()
    prediction = predict_label(metadata["sample_input"])
    assert prediction == metadata["expected_sample_prediction"]


def test_python_consumer_predicts_setosa_probability_using_contract_sample():
    metadata = _load_setosa_metadata()
    probability = predict_setosa_probability(metadata["sample_input"])
    assert 0.0 <= probability <= 1.0
    assert abs(probability - metadata["expected_sample_prediction"]) < 0.001


def test_python_consumer_predicts_label_using_contract_sample():
    metadata = _load_iris_metadata()
    prediction = predict_label(metadata["sample_input"])
    assert prediction == metadata["expected_sample_prediction"]


def test_python_consumer_predicts_setosa_probability_using_contract_sample():
    metadata = _load_setosa_metadata()
    probability = predict_setosa_probability(metadata["sample_input"])
    assert 0.0 <= probability <= 1.0
    assert abs(probability - metadata["expected_sample_prediction"]) < 0.001
