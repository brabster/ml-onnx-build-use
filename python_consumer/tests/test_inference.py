from time import perf_counter

from python_consumer.inference import predict_label, predict_setosa_probability


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
    features = [5.1, 3.5, 1.4, 0.2]
    runs = 100
    durations_ms = []
    predictions = []

    for _ in range(runs):
        started_at = perf_counter()
        predictions.append(predict_label(features))
        durations_ms.append((perf_counter() - started_at) * 1000)

    print(
        "predict_label latency over "
        f"{runs} runs: min={min(durations_ms):.3f}ms "
        f"avg={sum(durations_ms) / runs:.3f}ms max={max(durations_ms):.3f}ms"
    )
    assert all(prediction == predictions[0] for prediction in predictions)
