package com.example.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

record ModelContract(float[] sampleInput, double expectedSamplePrediction, int featureCount, double predictionTolerance) {
    static ModelContract load(Path metadataPath) throws IOException {
        String json = Files.readString(metadataPath);
        return new ModelContract(
                parseFloatArray(json, "sample_input"),
                parseDouble(json, "expected_sample_prediction"),
                (int) parseDouble(json, "feature_count"),
                parseDouble(json, "prediction_tolerance"));
    }

    private static double parseDouble(String json, String fieldName) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(fieldName) + "\"\\s*:\\s*([-0-9.Ee+]+)").matcher(json);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Missing numeric field: " + fieldName);
        }
        return Double.parseDouble(matcher.group(1));
    }

    private static float[] parseFloatArray(String json, String fieldName) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(fieldName) + "\"\\s*:\\s*\\[(.*?)]", Pattern.DOTALL)
                .matcher(json);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Missing array field: " + fieldName);
        }

        String[] parts = matcher.group(1).split(",");
        float[] values = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = Float.parseFloat(parts[i].trim());
        }
        return values;
    }
}
