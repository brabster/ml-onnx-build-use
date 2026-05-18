package com.example.onnx;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class IrisPredictorTest {

    @Test
    void predictsSetosaFromThePackagedModel() throws Exception {
        float[] input = {5.1f, 3.5f, 1.4f, 0.2f};
        try (IrisPredictor predictor = new IrisPredictor(modelPath())) {
            long prediction = predictor.predict(input);
            System.out.printf("predict_label(%s) -> %d (expected 0)%n", java.util.Arrays.toString(input), prediction);
            assertEquals(0L, prediction);
        }
    }

    @Test
    void reportsInferenceLatencyAndConsistencyExample() throws Exception {
        float[] input = {5.1f, 3.5f, 1.4f, 0.2f};
        int runs = 10;
        long[] durationsNs = new long[runs];
        long[] predictions = new long[runs];

        try (IrisPredictor predictor = new IrisPredictor(modelPath())) {
            for (int i = 0; i < runs; i++) {
                long startedAt = System.nanoTime();
                predictions[i] = predictor.predict(input);
                durationsNs[i] = System.nanoTime() - startedAt;
            }
        }

        long minNs = java.util.Arrays.stream(durationsNs).min().orElse(0L);
        long maxNs = java.util.Arrays.stream(durationsNs).max().orElse(0L);
        double avgNs = java.util.Arrays.stream(durationsNs).average().orElse(0.0);
        System.out.printf(
                "predict_label latency over %d runs: min=%.3fms avg=%.3fms max=%.3fms%n",
                runs,
                minNs / 1_000_000.0,
                avgNs / 1_000_000.0,
                maxNs / 1_000_000.0);

        for (long prediction : predictions) {
            assertEquals(predictions[0], prediction);
        }
    }

    private static Path modelPath() {
        String configuredPath = System.getProperty(
                "model.path",
                System.getenv().getOrDefault("MODEL_PATH", "../producer/dist/iris_classifier.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }
}
