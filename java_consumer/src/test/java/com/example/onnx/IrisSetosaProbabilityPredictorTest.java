package com.example.onnx;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class IrisSetosaProbabilityPredictorTest {

    @Test
    void predictsHighSetosaProbabilityForSetosaFromThePackagedModel() throws Exception {
        ModelContract contract = ModelContract.load(modelPath().resolveSibling("setosa_probability_metadata.json"));
        assertTrue(contract.featureCount() == 4, "the contract should require four input features");

        try (IrisSetosaProbabilityPredictor predictor = new IrisSetosaProbabilityPredictor(modelPath())) {
            float probability = predictor.predict(contract.sampleInput());
            assertTrue(probability >= 0.0f && probability <= 1.0f, "probability must be between 0 and 1");
            assertTrue(
                    Math.abs(probability - contract.expectedSamplePrediction()) <= contract.predictionTolerance(),
                    "prediction must match the producer contract tolerance");
        }
    }

    private static Path modelPath() {
        String configuredPath = System.getProperty(
                "setosa.model.path",
                System.getenv().getOrDefault("SETOSA_MODEL_PATH", "../producer/dist/setosa_probability.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }
}
