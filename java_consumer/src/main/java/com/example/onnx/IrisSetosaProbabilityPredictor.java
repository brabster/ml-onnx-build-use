package com.example.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.FloatBuffer;
import java.nio.file.Path;

public final class IrisSetosaProbabilityPredictor implements AutoCloseable {
    private final OrtEnvironment environment;
    private final OrtSession session;

    public IrisSetosaProbabilityPredictor(Path modelPath) throws OrtException {
        this.environment = OrtEnvironment.getEnvironment();
        this.session = environment.createSession(modelPath.toString(), new OrtSession.SessionOptions());
    }

    public float predict(float[] features) throws OrtException {
        if (features.length != 4) {
            throw new IllegalArgumentException("The iris model expects exactly four numeric features.");
        }

        try (OnnxTensor inputTensor =
                     OnnxTensor.createTensor(environment, FloatBuffer.wrap(features), new long[]{1, 4});
             OrtSession.Result result = session.run(java.util.Map.of("features", inputTensor))) {
            Object probabilityValue = result.get(0).getValue();
            if (probabilityValue instanceof float[] probabilities) {
                return probabilities[0];
            }
            throw new IllegalStateException("Unexpected ONNX probability output type: " + probabilityValue.getClass());
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
