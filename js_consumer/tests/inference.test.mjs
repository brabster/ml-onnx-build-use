import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { predictValue } from '../inference.mjs';

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

test('predicts a regression value from the packaged model', async () => {
    const features = [
        0.5136836171150208, -0.6633052825927734, -0.40696072578430176, 0.9416861534118652,
        0.08007215708494186, -0.7074018716812134, -1.4520694017410278, -0.0969497561454773,
        0.2586694657802582, -1.6983729600906372,
    ];
    const result = await predictValue(features);
    console.log(`predictValue(features) -> ${result.toFixed(4)}`);
    assert.ok(Math.abs(result - (-145.16)) < 1.0, `Expected prediction close to -145.16, got ${result}`);
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
        await predictValue(warmupFeatures);
    }

    for (let i = 0; i < measuredRuns; i += 1) {
        const features = samples[i % samples.length];
        const startedAt = performance.now();
        const prediction = await predictValue(features);
        durationsMs.push(performance.now() - startedAt);
        const sampleKey = features.join(',');
        if (firstPredictionBySample.has(sampleKey)) {
            assert.ok(
                Math.abs(prediction - firstPredictionBySample.get(sampleKey)) < 0.001,
                `Inconsistent prediction for sample ${sampleKey}`,
            );
        } else {
            firstPredictionBySample.set(sampleKey, prediction);
        }
    }

    const sortedMs = [...durationsMs].sort((a, b) => a - b);
    const p50Ms = sortedMs[Math.floor(0.50 * measuredRuns)];
    const p95Ms = sortedMs[Math.floor(0.95 * measuredRuns)];
    const p99Ms = sortedMs[Math.floor(0.99 * measuredRuns)];
    const maxMs = sortedMs[measuredRuns - 1];
    console.log(
        `predictValue latency over ${measuredRuns} measured runs after ${warmupRuns} warmup runs: P50=${p50Ms.toFixed(3)}ms P95=${p95Ms.toFixed(3)}ms P99=${p99Ms.toFixed(3)}ms max=${maxMs.toFixed(3)}ms across ${samples.length} committed sample inputs`,
    );
});
