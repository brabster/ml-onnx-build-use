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
