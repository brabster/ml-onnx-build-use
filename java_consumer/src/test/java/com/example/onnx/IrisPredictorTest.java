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
        List<float[]> samples = loadLatencyInputSamples();
        assertFalse(samples.isEmpty());
        int warmupRuns = 10;
        int measuredRuns = 1000;
        long[] durationsNs = new long[measuredRuns];
        Map<String, Long> firstPredictionBySample = new HashMap<>();

        try (IrisPredictor predictor = new IrisPredictor(modelPath())) {
            for (int warmupRun = 0; warmupRun < warmupRuns; warmupRun++) {
                float[] warmupInput = samples.get(warmupRun % samples.size());
                predictor.predict(warmupInput);
            }

            for (int i = 0; i < measuredRuns; i++) {
                float[] input = samples.get(i % samples.size());
                long startedAt = System.nanoTime();
                long prediction = predictor.predict(input);
                durationsNs[i] = System.nanoTime() - startedAt;
                String sampleKey = java.util.Arrays.toString(input);
                if (firstPredictionBySample.containsKey(sampleKey)) {
                    assertEquals(firstPredictionBySample.get(sampleKey), prediction);
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
                "predict_label latency over %d measured runs after %d warmup runs: P50=%.3fms P95=%.3fms P99=%.3fms max=%.3fms across %d committed sample inputs%n",
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
                System.getenv().getOrDefault("MODEL_PATH", "../producer/dist/iris_classifier.onnx"));
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
            if (values.length != 4) {
                throw new IllegalStateException("Expected 4 float values in sample row: " + line);
            }
            float[] sample = new float[4];
            for (int i = 0; i < values.length; i++) {
                sample[i] = Float.parseFloat(values[i]);
            }
            samples.add(sample);
        }
        return samples;
    }
}
