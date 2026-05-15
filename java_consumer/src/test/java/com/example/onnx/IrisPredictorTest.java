package com.example.onnx;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class IrisPredictorTest {

    @Test
    void predictsSetosaFromThePackagedModel() throws Exception {
        ModelContract contract = ModelContract.load(modelPath().resolveSibling("model_metadata.json"));
        assertEquals(4, contract.featureCount());

        try (IrisPredictor predictor = new IrisPredictor(modelPath())) {
            long prediction = predictor.predict(contract.sampleInput());
            assertEquals(Math.round(contract.expectedSamplePrediction()), prediction);
        }
    }

    private static Path modelPath() {
        String configuredPath = System.getProperty(
                "model.path",
                System.getenv().getOrDefault("MODEL_PATH", "../producer/dist/iris_classifier.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }
}
