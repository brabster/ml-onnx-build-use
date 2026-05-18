import csv
from pathlib import Path
from time import perf_counter

import pytest

from python_consumer.inference import (
    predict_value,
    predict_value_from_pickle,
)


LATENCY_INPUT_SAMPLES_PATH = (
    Path(__file__).resolve().parents[2] / "producer" / "contracts" / "latency_input_samples.csv"
)


def load_latency_input_samples() -> list[list[float]]:
    with LATENCY_INPUT_SAMPLES_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return [[float(value) for value in row] for row in reader if row]


@pytest.mark.onnx
def test_python_consumer_predicts_a_value_from_the_packaged_model():
    features = [0.5136836171150208, -0.6633052825927734, -0.40696072578430176, 0.9416861534118652,
                0.08007215708494186, -0.7074018716812134, -1.4520694017410278, -0.0969497561454773,
                0.2586694657802582, -1.6983729600906372]
    result = predict_value(features)
    print(f"predict_value(features) -> {result:.4f}")
    assert abs(result - (-145.16)) < 1.0


@pytest.mark.pickle
def test_python_consumer_predicts_a_value_from_the_packaged_pickle_model():
    features = [0.5136836171150208, -0.6633052825927734, -0.40696072578430176, 0.9416861534118652,
                0.08007215708494186, -0.7074018716812134, -1.4520694017410278, -0.0969497561454773,
                0.2586694657802582, -1.6983729600906372]
    result = predict_value_from_pickle(features)
    print(f"predict_value_from_pickle(features) -> {result:.4f}")
    assert abs(result - (-145.16)) < 1.0


@pytest.mark.onnx
def test_python_consumer_onnx_inference_latency_and_consistency_example():
    samples = load_latency_input_samples()
    assert samples
    warmup_runs = 10
    measured_runs = 1000
    durations_ms = []
    first_prediction_by_sample = {}

    for warmup_run in range(warmup_runs):
        warmup_features = samples[warmup_run % len(samples)]
        predict_value(warmup_features)

    for run in range(measured_runs):
        features = samples[run % len(samples)]
        started_at = perf_counter()
        prediction = predict_value(features)
        durations_ms.append((perf_counter() - started_at) * 1000)
        sample_key = tuple(features)
        if sample_key in first_prediction_by_sample:
            assert abs(prediction - first_prediction_by_sample[sample_key]) < 0.001
        else:
            first_prediction_by_sample[sample_key] = prediction

    sorted_durations = sorted(durations_ms)
    p50 = sorted_durations[int(0.50 * measured_runs)]
    p95 = sorted_durations[int(0.95 * measured_runs)]
    p99 = sorted_durations[int(0.99 * measured_runs)]
    print(
        "predict_value latency over "
        f"{measured_runs} measured runs after {warmup_runs} warmup runs: "
        f"P50={p50:.3f}ms "
        f"P95={p95:.3f}ms "
        f"P99={p99:.3f}ms "
        f"max={sorted_durations[-1]:.3f}ms across {len(samples)} committed sample inputs"
    )


@pytest.mark.pickle
def test_python_consumer_pickle_inference_latency_and_consistency_example():
    samples = load_latency_input_samples()
    assert samples
    warmup_runs = 10
    measured_runs = 1000
    durations_ms = []
    first_prediction_by_sample = {}

    for warmup_run in range(warmup_runs):
        warmup_features = samples[warmup_run % len(samples)]
        predict_value_from_pickle(warmup_features)

    for run in range(measured_runs):
        features = samples[run % len(samples)]
        started_at = perf_counter()
        prediction = predict_value_from_pickle(features)
        durations_ms.append((perf_counter() - started_at) * 1000)
        sample_key = tuple(features)
        if sample_key in first_prediction_by_sample:
            assert abs(prediction - first_prediction_by_sample[sample_key]) < 0.001
        else:
            first_prediction_by_sample[sample_key] = prediction

    sorted_durations = sorted(durations_ms)
    p50 = sorted_durations[int(0.50 * measured_runs)]
    p95 = sorted_durations[int(0.95 * measured_runs)]
    p99 = sorted_durations[int(0.99 * measured_runs)]
    print(
        "predict_value_from_pickle latency over "
        f"{measured_runs} measured runs after {warmup_runs} warmup runs: "
        f"P50={p50:.3f}ms "
        f"P95={p95:.3f}ms "
        f"P99={p99:.3f}ms "
        f"max={sorted_durations[-1]:.3f}ms across {len(samples)} committed sample inputs"
    )
