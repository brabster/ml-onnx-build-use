import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { predictLabel, predictSetosaProbability } from '../inference.mjs';

async function loadLatencyInputSamples() {
    const content = await readFile(
        new URL('../../producer/contracts/latency_input_samples.csv', import.meta.url),
        'utf8',
    );
    const lines = content
        .trim()
        .split('\n')
        .slice(1)
        .filter((line) => line.trim().length > 0);
    return lines.map((line) => line.split(',').map((value) => Number.parseFloat(value)));
}

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
    const samples = await loadLatencyInputSamples();
    assert.ok(samples.length > 0);
    const warmupRuns = 10;
    const measuredRuns = 1000;
    const durationsMs = [];
    const firstPredictionBySample = new Map();

    for (let warmupRun = 0; warmupRun < warmupRuns; warmupRun += 1) {
        const warmupFeatures = samples[warmupRun % samples.length];
        await predictLabel(warmupFeatures);
    }

    for (let i = 0; i < measuredRuns; i += 1) {
        const features = samples[i % samples.length];
        const startedAt = performance.now();
        const prediction = await predictLabel(features);
        durationsMs.push(performance.now() - startedAt);
        const sampleKey = features.join(',');
        if (firstPredictionBySample.has(sampleKey)) {
            assert.strictEqual(prediction, firstPredictionBySample.get(sampleKey));
        } else {
            firstPredictionBySample.set(sampleKey, prediction);
        }
    }

    const minMs = Math.min(...durationsMs);
    const maxMs = Math.max(...durationsMs);
    const avgMs = durationsMs.reduce((sum, value) => sum + value, 0) / measuredRuns;
    console.log(
        `predictLabel latency over ${measuredRuns} measured runs after ${warmupRuns} warmup runs: min=${minMs.toFixed(3)}ms avg=${avgMs.toFixed(3)}ms max=${maxMs.toFixed(3)}ms across ${samples.length} committed sample inputs`,
    );
});
