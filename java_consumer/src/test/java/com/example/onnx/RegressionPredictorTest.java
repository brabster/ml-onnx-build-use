package com.example.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RegressionPredictorTest {

    @Test
    void predictsAValueFromThePackagedModel() throws Exception {
        float[] input = {
            0.5136836f, -0.6633053f, -0.4069607f, 0.9416862f, 0.0800722f,
            -0.7074019f, -1.4520694f, -0.0969498f, 0.2586695f, -1.6983730f
        };
        try (RegressionPredictor predictor = new RegressionPredictor(modelPath())) {
            float prediction = predictor.predict(input);
            System.out.printf("predict_value(features) -> %.4f (expected ~-145.16)%n", prediction);
            assertTrue(Math.abs(prediction - (-145.16f)) < 1.0f,
                    "Expected prediction close to -145.16, got " + prediction);
        }
    }

    @Test
    void reportsInferenceLatencyAndConsistencyExample() throws Exception {
        List<float[]> samples = loadLatencyInputSamples();
        assertFalse(samples.isEmpty());
        int warmupRuns = 10;
        int measuredRuns = 1000;
        long[] durationsNs = new long[measuredRuns];
        Map<String, Float> firstPredictionBySample = new HashMap<>();

        try (RegressionPredictor predictor = new RegressionPredictor(modelPath())) {
            for (int warmupRun = 0; warmupRun < warmupRuns; warmupRun++) {
                float[] warmupInput = samples.get(warmupRun % samples.size());
                predictor.predict(warmupInput);
            }

            for (int i = 0; i < measuredRuns; i++) {
                float[] input = samples.get(i % samples.size());
                long startedAt = System.nanoTime();
                float prediction = predictor.predict(input);
                durationsNs[i] = System.nanoTime() - startedAt;
                String sampleKey = java.util.Arrays.toString(input);
                if (firstPredictionBySample.containsKey(sampleKey)) {
                    assertEquals(firstPredictionBySample.get(sampleKey), prediction, 0.001f);
                } else {
                    firstPredictionBySample.put(sampleKey, prediction);
                }
            }
        }

        java.util.Arrays.sort(durationsNs);
        long p50Ns = durationsNs[(int) (0.50 * measuredRuns)];
        long p95Ns = durationsNs[(int) (0.95 * measuredRuns)];
        long p99Ns = durationsNs[(int) (0.99 * measuredRuns)];
        long maxNs = durationsNs[measuredRuns - 1];
        System.out.printf(
                "predict_value latency over %d measured runs after %d warmup runs: P50=%.3fms P95=%.3fms P99=%.3fms max=%.3fms across %d committed sample inputs%n",
                measuredRuns,
                warmupRuns,
                p50Ns / 1_000_000.0,
                p95Ns / 1_000_000.0,
                p99Ns / 1_000_000.0,
                maxNs / 1_000_000.0,
                samples.size());
    }

    private static Path modelPath() {
        String configuredPath = System.getProperty(
                "model.path",
                System.getenv().getOrDefault("MODEL_PATH", "../producer/dist/regression_model.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }

    private static Path latencyInputSamplesPath() {
        String configuredPath = System.getProperty(
                "latency.input.samples.path",
                System.getenv().getOrDefault(
                        "LATENCY_INPUT_SAMPLES_PATH",
                        "../producer/contracts/latency_input_samples.csv"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }

    private static List<float[]> loadLatencyInputSamples() throws IOException {
        List<float[]> samples = new ArrayList<>();
        List<String> lines = Files.readAllLines(latencyInputSamplesPath());
        for (int lineIndex = 1; lineIndex < lines.size(); lineIndex++) {
            String line = lines.get(lineIndex).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] values = line.split(",");
            if (values.length != RegressionPredictor.N_FEATURES) {
                throw new IllegalStateException(
                        "Expected " + RegressionPredictor.N_FEATURES + " float values in sample row: " + line);
            }
            float[] sample = new float[RegressionPredictor.N_FEATURES];
            for (int i = 0; i < values.length; i++) {
                sample[i] = Float.parseFloat(values[i]);
            }
            samples.add(sample);
        }
        return samples;
    }
}
