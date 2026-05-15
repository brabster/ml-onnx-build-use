package com.example.onnx;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class IrisPredictorTest {

    // Metadata filename is the contract: both producer and consumer must agree on this name.
    private static final String METADATA_FILE = "model_metadata.json";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void predictsLabelUsingContractSample() throws Exception {
        Path metadataPath = modelPath().getParent().resolve(METADATA_FILE);
        JsonNode metadata = MAPPER.readTree(metadataPath.toFile());

        float[] sampleInput = toFloatArray(metadata.get("sample_input"));
        long expectedPrediction = metadata.get("expected_sample_prediction").longValue();

        try (IrisPredictor predictor = new IrisPredictor(modelPath())) {
            assertEquals(expectedPrediction, predictor.predict(sampleInput));
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
                "model.path",
                System.getenv().getOrDefault("MODEL_PATH", "../producer/dist/iris_classifier.onnx"));
        return Paths.get(configuredPath).toAbsolutePath().normalize();
    }
}
