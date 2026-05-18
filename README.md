# ml-onnx-build-use

This repository is a small, plain-English example of one ONNX model being produced once and consumed in several different runtimes.

## What is in the repo?

- `producer/` trains a model with scikit-learn and exports ONNX and pickle files.
  - `train_and_package.py` — synthetic regression model trained on 100,000 samples generated with `make_regression` (random_state=42), exported as `regression_model.onnx` and `regression_model.pkl`.
  - `contracts/latency_input_samples.csv` — committed random regression feature samples shared by all consumer latency tests.
- `python_consumer/` loads the ONNX file with ONNX Runtime for Python, also loads the pickled model, and checks inferences in tests.
- `java_consumer/` loads the same ONNX file with ONNX Runtime for Java and checks the same inferences in tests.
- `js_consumer/` loads the same ONNX file with ONNX Runtime Web and checks the same inferences in tests, using the WebAssembly backend that also powers in-browser inference.
- `.github/workflows/` contains three reusable pipelines plus one orchestration workflow. The producer pipeline publishes the packaged models as a GitHub Actions artifact. The three consumer pipelines download that artifact and run their tests against it.

## What were the consumer results?

The consumers were able to perform inference using the ONNX models, and got the correct answers. Performance of the inferences, measured after a warm up set and over 1000 inferences (looping over the same 15 randomly-generated inputs) produced the following metrics in [this CI run](https://github.com/brabster/ml-onnx-build-use/actions/runs/26023703853). The Python consumer workflow now prints ONNX and pickle latency metrics in separate CI tasks.

|Consumer|P50(ms)|P95(ms)|P99(ms)|max(ms)|
|--------|-------|-------|-------|-------|
|Python|0.63|0.70|2.61|6.00|
|Java|0.03|0.06|0.22|8.07|
|JavaScript (approximating in-browser inference)|0.95|4.21|6.40|12.56|

The numbers will vary in real implementations.

## Why a synthetic regression dataset?

The synthetic regression dataset is generated with scikit-learn's `make_regression` (100,000 samples, 10 features, random_state=42). It is orders of magnitude larger than the classic iris dataset (150 samples), making it more representative of real-world model training scale while remaining fully reproducible.

## Run the example locally

1. Train and package the models.

   ```bash
   python3 -m pip install -r producer/requirements.txt
   python3 producer/train_and_package.py --output-dir producer/dist
   ```

2. Run the Python consumer tests.

   ```bash
   python3 -m pip install -r python_consumer/requirements.txt
   MODEL_PATH=producer/dist/regression_model.onnx \
     PICKLE_MODEL_PATH=producer/dist/regression_model.pkl \
     python3 -m pytest python_consumer/tests
   ```

3. Run the Java consumer tests.

   ```bash
   cd java_consumer
   MODEL_PATH=../producer/dist/regression_model.onnx \
     mvn --batch-mode test
   ```

4. Run the JavaScript consumer tests.

   ```bash
   cd js_consumer
   npm install
   MODEL_PATH=../producer/dist/regression_model.onnx \
     npm test
   ```

## GitHub Actions flow

1. `ci.yml` runs on pushes and pull requests, then calls the four reusable workflows in order.
2. `producer.yml` trains the model, tests the exporter, and uploads `producer/dist` (ONNX and pickle artifacts) as the `packaged-onnx-model` artifact.
3. `python-consumer.yml` downloads that artifact and runs the Python inference tests in two CI tasks: one for ONNX and one for pickle (each including latency/consistency examples over shared committed sample inputs with 10 warmup runs and 1000 measured runs).
4. `java-consumer.yml` does the same for the Java inference tests (including latency/consistency examples over shared committed sample inputs with 10 warmup runs and 1000 measured runs).
5. `js-consumer.yml` does the same for the JavaScript inference tests (including latency/consistency examples over shared committed sample inputs with 10 warmup runs and 1000 measured runs).

The end result is one model package and three independent consumers proving that it can be used from Python, Java, and JavaScript (including browsers via WebAssembly).

## Note about ONNX Runtime PCI warning in GitHub Actions

You may see this warning in Linux-based GitHub Actions logs:

`[W:onnxruntime:..., device_discovery.cc:133 GetPciBusId] Skipping pci_bus_id ... did not match expected pattern ...`

This comes from ONNX Runtime probing host hardware paths on hosted runners. It is expected in this environment and does not indicate a model or test failure.
