package com.example.onnx;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class IrisSetosaProbabilityPredictorTest {

    // Metadata filename is the contract: both producer and consumer must agree on this name.
    private static final String METADATA_FILE = "setosa_probability_metadata.json";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void predictsProbabilityUsingContractSample() throws Exception {
        Path metadataPath = modelPath().getParent().resolve(METADATA_FILE);
        JsonNode metadata = MAPPER.readTree(metadataPath.toFile());

        float[] sampleInput = toFloatArray(metadata.get("sample_input"));
        float expectedPrediction = (float) metadata.get("expected_sample_prediction").doubleValue();

        try (IrisSetosaProbabilityPredictor predictor = new IrisSetosaProbabilityPredictor(modelPath())) {
            float probability = predictor.predict(sampleInput);
            assertTrue(probability >= 0.0f && probability <= 1.0f, "probability must be between 0 and 1");
            assertTrue(Math.abs(probability - expectedPrediction) < 0.001f,
                    "probability should match the contract sample expectation");
        }
    }

    private static float[] toFloatArray(JsonNode arrayNode) {
        float[] values = new float[arrayNode.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) arrayNode.get(i).doubleValue();
        }
        return values;
    }

    private static Path modelPath() {
        String configuredPath = System.getProperty(
                "setosa.model.path",
                System.getenv().getOrDefault("SETOSA_MODEL_PATH", "../producer/dist/setosa_probability.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }
}
