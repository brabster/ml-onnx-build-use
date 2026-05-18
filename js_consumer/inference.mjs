import * as ort from 'onnxruntime-web';
import { fileURLToPath } from 'url';
import { dirname, join, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Point the WASM backend at the bundled .wasm files so this module works in Node.js
// the same way onnxruntime-web works in a browser (via WebAssembly).
ort.env.wasm.wasmPaths = join(__dirname, 'node_modules', 'onnxruntime-web', 'dist') + '/';

const DEFAULT_MODEL_PATH = resolve(__dirname, '..', 'producer', 'dist', 'regression_model.onnx');
const N_FEATURES = 10;

export async function predictValue(features, modelPath) {
    if (features.length !== N_FEATURES) {
        throw new Error(`The regression model expects exactly ${N_FEATURES} numeric features.`);
    }
    const path = modelPath ?? process.env.MODEL_PATH ?? DEFAULT_MODEL_PATH;
    const session = await ort.InferenceSession.create(path);
    const tensor = new ort.Tensor('float32', new Float32Array(features), [1, N_FEATURES]);
    const results = await session.run({ features: tensor });
    return Number(results.variable.data[0]);
}
