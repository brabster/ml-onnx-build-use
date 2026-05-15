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

    private static Path modelPath() {
        String configuredPath = System.getProperty(
                "model.path",
                System.getenv().getOrDefault("MODEL_PATH", "../producer/dist/iris_classifier.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }
}
