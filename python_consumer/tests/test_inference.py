from python_consumer.inference import predict_label


def test_python_consumer_predicts_setosa_from_the_packaged_model():
    assert predict_label([5.1, 3.5, 1.4, 0.2]) == 0
