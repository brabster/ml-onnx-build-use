import json
from pathlib import Path

from python_consumer.inference import (
    predict_label,
    predict_setosa_probability,
    resolve_model_path,
    resolve_probability_model_path,
)


def _load_contract(metadata_path: Path) -> dict[str, object]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def test_python_consumer_predicts_setosa_from_the_packaged_model():
    contract = _load_contract(resolve_model_path().with_name("model_metadata.json"))

    sample_input = contract["sample_input"]
    assert isinstance(sample_input, list)
    assert len(sample_input) == contract["feature_count"]
    assert predict_label(sample_input) == contract["expected_sample_prediction"]


def test_python_consumer_predicts_setosa_probability_from_the_packaged_model():
    contract = _load_contract(resolve_probability_model_path().with_name("setosa_probability_metadata.json"))

    sample_input = contract["sample_input"]
    assert isinstance(sample_input, list)
    assert len(sample_input) == contract["feature_count"]
    probability = predict_setosa_probability(sample_input)
    assert 0.0 <= probability <= 1.0
    assert abs(probability - contract["expected_sample_prediction"]) <= contract["prediction_tolerance"]
