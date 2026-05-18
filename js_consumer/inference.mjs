import * as ort from 'onnxruntime-web';
import { fileURLToPath } from 'url';
import { dirname, join, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Point the WASM backend at the bundled .wasm files so this module works in Node.js
// the same way onnxruntime-web works in a browser (via WebAssembly).
ort.env.wasm.wasmPaths = join(__dirname, 'node_modules', 'onnxruntime-web', 'dist') + '/';

const DEFAULT_MODEL_PATH = resolve(__dirname, '..', 'producer', 'dist', 'iris_classifier.onnx');
const DEFAULT_SETOSA_MODEL_PATH = resolve(__dirname, '..', 'producer', 'dist', 'setosa_probability.onnx');

export async function predictLabel(features, modelPath) {
    if (features.length !== 4) {
        throw new Error('The iris model expects exactly four numeric features.');
    }
    const path = modelPath ?? process.env.MODEL_PATH ?? DEFAULT_MODEL_PATH;
    const session = await ort.InferenceSession.create(path);
    const tensor = new ort.Tensor('float32', new Float32Array(features), [1, 4]);
    const results = await session.run({ features: tensor });
    return Number(results.label.data[0]);
}

export async function predictSetosaProbability(features, modelPath) {
    if (features.length !== 4) {
        throw new Error('The iris model expects exactly four numeric features.');
    }
    const path = modelPath ?? process.env.SETOSA_MODEL_PATH ?? DEFAULT_SETOSA_MODEL_PATH;
    const session = await ort.InferenceSession.create(path);
    const tensor = new ort.Tensor('float32', new Float32Array(features), [1, 4]);
    const results = await session.run({ features: tensor });
    return results.setosa_probability.data[0];
}
