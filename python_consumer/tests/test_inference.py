from python_consumer.inference import predict_label, predict_setosa_probability


def test_python_consumer_predicts_setosa_from_the_packaged_model():
    assert predict_label([5.1, 3.5, 1.4, 0.2]) == 0


def test_python_consumer_predicts_setosa_probability_from_the_packaged_model():
    probability = predict_setosa_probability([5.1, 3.5, 1.4, 0.2])
    assert 0.0 <= probability <= 1.0
    assert probability > 0.9
