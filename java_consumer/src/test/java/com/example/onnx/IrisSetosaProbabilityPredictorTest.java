package com.example.onnx;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class IrisSetosaProbabilityPredictorTest {

    @Test
    void predictsHighSetosaProbabilityForSetosaFromThePackagedModel() throws Exception {
        try (IrisSetosaProbabilityPredictor predictor = new IrisSetosaProbabilityPredictor(modelPath())) {
            float probability = predictor.predict(new float[]{5.1f, 3.5f, 1.4f, 0.2f});
            assertTrue(probability >= 0.0f && probability <= 1.0f, "probability must be between 0 and 1");
            assertTrue(probability > 0.9f, "a known Setosa sample should have high Setosa probability");
        }
    }

    private static Path modelPath() {
        String configuredPath = System.getProperty(
                "setosa.model.path",
                System.getenv().getOrDefault("SETOSA_MODEL_PATH", "../producer/dist/setosa_probability.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }
}
