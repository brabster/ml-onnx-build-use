package com.example.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.FloatBuffer;
import java.nio.file.Path;

public final class RegressionPredictor implements AutoCloseable {
    public static final int N_FEATURES = 10;

    private final OrtEnvironment environment;
    private final OrtSession session;

    public RegressionPredictor(Path modelPath) throws OrtException {
        this.environment = OrtEnvironment.getEnvironment();
        this.session = environment.createSession(modelPath.toString(), new OrtSession.SessionOptions());
    }

    public float predict(float[] features) throws OrtException {
        if (features.length != N_FEATURES) {
            throw new IllegalArgumentException(
                    "The regression model expects exactly " + N_FEATURES + " numeric features.");
        }

        try (OnnxTensor inputTensor =
                     OnnxTensor.createTensor(environment, FloatBuffer.wrap(features), new long[]{1, N_FEATURES});
             OrtSession.Result result = session.run(java.util.Map.of("features", inputTensor))) {
            Object value = result.get(0).getValue();
            if (value instanceof float[][] matrix) {
                return matrix[0][0];
            }
            if (value instanceof float[] array) {
                return array[0];
            }
            throw new IllegalStateException("Unexpected ONNX output type: " + value.getClass());
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
