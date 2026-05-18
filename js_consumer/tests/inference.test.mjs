import { test } from 'node:test';
import assert from 'node:assert/strict';
import { predictLabel, predictSetosaProbability } from '../inference.mjs';

test('predicts Setosa class label from the packaged model', async () => {
    const features = [5.1, 3.5, 1.4, 0.2];
    const result = await predictLabel(features);
    console.log(`predictLabel([5.1, 3.5, 1.4, 0.2]) -> ${result}`);
    assert.strictEqual(result, 0);
});

test('predicts high Setosa probability from the packaged model', async () => {
    const features = [5.1, 3.5, 1.4, 0.2];
    const probability = await predictSetosaProbability(features);
    console.log(`predictSetosaProbability([5.1, 3.5, 1.4, 0.2]) -> ${probability.toFixed(4)}`);
    assert.ok(probability >= 0.0 && probability <= 1.0, 'probability must be between 0 and 1');
    assert.ok(probability > 0.9, 'a known Setosa sample should have high Setosa probability');
});

test('reports inference latency and consistency example', async () => {
    const features = [5.1, 3.5, 1.4, 0.2];
    const runs = 10;
    const durationsMs = [];
    const predictions = [];

    for (let i = 0; i < runs; i += 1) {
        const startedAt = performance.now();
        predictions.push(await predictLabel(features));
        durationsMs.push(performance.now() - startedAt);
    }

    const minMs = Math.min(...durationsMs);
    const maxMs = Math.max(...durationsMs);
    const avgMs = durationsMs.reduce((sum, value) => sum + value, 0) / runs;
    console.log(
        `predictLabel latency over ${runs} runs: min=${minMs.toFixed(3)}ms avg=${avgMs.toFixed(3)}ms max=${maxMs.toFixed(3)}ms`,
    );

    assert.ok(predictions.every((prediction) => prediction === predictions[0]));
});
