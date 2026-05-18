import csv
from pathlib import Path
from time import perf_counter

from python_consumer.inference import predict_label, predict_setosa_probability


LATENCY_INPUT_SAMPLES_PATH = (
    Path(__file__).resolve().parents[2] / "producer" / "contracts" / "latency_input_samples.csv"
)


def load_latency_input_samples() -> list[list[float]]:
    with LATENCY_INPUT_SAMPLES_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return [[float(value) for value in row] for row in reader if row]


def test_python_consumer_predicts_setosa_from_the_packaged_model():
    result = predict_label([5.1, 3.5, 1.4, 0.2])
    print(f"predict_label([5.1, 3.5, 1.4, 0.2]) -> {result}")
    assert result == 0


def test_python_consumer_predicts_setosa_probability_from_the_packaged_model():
    probability = predict_setosa_probability([5.1, 3.5, 1.4, 0.2])
    print(f"predict_setosa_probability([5.1, 3.5, 1.4, 0.2]) -> {probability:.4f}")
    assert 0.0 <= probability <= 1.0
    assert probability > 0.9


def test_python_consumer_inference_latency_and_consistency_example():
    samples = load_latency_input_samples()
    assert samples
    runs = 100
    durations_ms = []
    first_prediction_by_sample = {}

    for run in range(runs):
        features = samples[run % len(samples)]
        started_at = perf_counter()
        prediction = predict_label(features)
        durations_ms.append((perf_counter() - started_at) * 1000)
        sample_key = tuple(features)
        if sample_key in first_prediction_by_sample:
            assert prediction == first_prediction_by_sample[sample_key]
        else:
            first_prediction_by_sample[sample_key] = prediction

    print(
        "predict_label latency over "
        f"{runs} runs: min={min(durations_ms):.3f}ms "
        f"avg={sum(durations_ms) / runs:.3f}ms max={max(durations_ms):.3f}ms "
        f"across {len(samples)} committed sample inputs"
    )
