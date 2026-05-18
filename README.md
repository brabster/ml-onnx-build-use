# ml-onnx-build-use

This repository is a small, plain-English example of one ONNX model being produced once and consumed in several different runtimes.

## What is in the repo?

- `producer/` trains two models with scikit-learn and exports ONNX and pickle files.
  - `train_and_package.py` — iris flower classifier that outputs a **long** class label (0, 1, or 2), exported as `iris_classifier.onnx` and `iris_classifier.pkl`.
  - `train_and_package_probability.py` — binary classifier that outputs a **float** probability of a flower being Iris Setosa (a value between 0 and 1).
  - `contracts/latency_input_samples.csv` — committed random Iris feature samples shared by all consumer latency tests.
- `python_consumer/` loads both ONNX files with ONNX Runtime for Python, also loads the pickled classifier, and checks inferences in tests (including ONNX vs pickle latency metrics for the classifier).
- `java_consumer/` loads the same ONNX files with ONNX Runtime for Java and checks the same inferences in tests.
- `js_consumer/` loads the same ONNX files with ONNX Runtime Web and checks the same inferences in tests, using the WebAssembly backend that also powers in-browser inference.
- `.github/workflows/` contains three reusable pipelines plus one orchestration workflow. The producer pipeline publishes the packaged models as a GitHub Actions artifact. The three consumer pipelines download that artifact and run their tests against it.

## What were the consumer results?

The consumers were able to perform inference using the ONNX models, and got the correct answers. Performance of the inferences, measured after a warm up set and over 1000 inferences (looping over the same 15 randomly-generated inputs) produced the following metrics in [this CI run](https://github.com/brabster/ml-onnx-build-use/actions/runs/26023703853). The Python consumer workflow now also prints comparable latency metrics for the pickled classifier artifact.

|Consumer|P50(ms)|P95(ms)|P99(ms)|max(ms)|
|--------|-------|-------|-------|-------|
|Python|0.63|0.70|2.61|6.00|
|Java|0.03|0.06|0.22|8.07|
|JavaScript (approximating in-browser inference)|0.95|4.21|6.40|12.56|

The numbers will vary in real implementations.

## Why the iris dataset?

The iris dataset ships with scikit-learn, is public, and is small enough to keep the example focused on packaging and inference instead of data wrangling.

## Run the example locally

1. Train and package the models.

   ```bash
   python3 -m pip install -r producer/requirements.txt
   python3 producer/train_and_package.py --output-dir producer/dist
   python3 producer/train_and_package_probability.py --output-dir producer/dist
   ```

2. Run the Python consumer tests.

   ```bash
   python3 -m pip install -r python_consumer/requirements.txt
   MODEL_PATH=producer/dist/iris_classifier.onnx \
     PICKLE_MODEL_PATH=producer/dist/iris_classifier.pkl \
     SETOSA_MODEL_PATH=producer/dist/setosa_probability.onnx \
     python3 -m pytest python_consumer/tests
   ```

3. Run the Java consumer tests.

   ```bash
   cd java_consumer
   MODEL_PATH=../producer/dist/iris_classifier.onnx \
     SETOSA_MODEL_PATH=../producer/dist/setosa_probability.onnx \
     mvn --batch-mode test
   ```

4. Run the JavaScript consumer tests.

   ```bash
   cd js_consumer
   npm install
   MODEL_PATH=../producer/dist/iris_classifier.onnx \
     SETOSA_MODEL_PATH=../producer/dist/setosa_probability.onnx \
     npm test
   ```

## GitHub Actions flow

1. `ci.yml` runs on pushes and pull requests, then calls the four reusable workflows in order.
2. `producer.yml` trains the model, tests the exporter, and uploads `producer/dist` (ONNX and pickle artifacts) as the `packaged-onnx-model` artifact.
3. `python-consumer.yml` downloads that artifact and runs the Python inference tests (including latency/consistency examples over shared committed sample inputs with 10 warmup runs and 1000 measured runs, reporting ONNX and pickle classifier latency metrics).
4. `java-consumer.yml` does the same for the Java inference tests (including latency/consistency examples over shared committed sample inputs with 10 warmup runs and 1000 measured runs).
5. `js-consumer.yml` does the same for the JavaScript inference tests (including latency/consistency examples over shared committed sample inputs with 10 warmup runs and 1000 measured runs).

The end result is one model package and three independent consumers proving that it can be used from Python, Java, and JavaScript (including browsers via WebAssembly).

## Note about ONNX Runtime PCI warning in GitHub Actions

You may see this warning in Linux-based GitHub Actions logs:

`[W:onnxruntime:..., device_discovery.cc:133 GetPciBusId] Skipping pci_bus_id ... did not match expected pattern ...`

This comes from ONNX Runtime probing host hardware paths on hosted runners. It is expected in this environment and does not indicate a model or test failure.
