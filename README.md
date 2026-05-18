# ml-onnx-build-use

This repository is a small, plain-English example of one ONNX model being produced once and consumed in two different runtimes.

## What is in the repo?

- `producer/` trains two models with scikit-learn and exports them as ONNX files.
  - `train_and_package.py` — iris flower classifier that outputs a **long** class label (0, 1, or 2).
  - `train_and_package_probability.py` — binary classifier that outputs a **float** probability of a flower being Iris Setosa (a value between 0 and 1).
- `python_consumer/` loads both ONNX files with ONNX Runtime for Python and checks inferences in tests.
- `java_consumer/` loads the same ONNX files with ONNX Runtime for Java and checks the same inferences in tests.
- `js_consumer/` loads the same ONNX files with ONNX Runtime Web and checks the same inferences in tests, using the WebAssembly backend that also powers in-browser inference.
- `.github/workflows/` contains three reusable pipelines plus one orchestration workflow. The producer pipeline publishes the packaged models as a GitHub Actions artifact. The three consumer pipelines download that artifact and run their tests against it.

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
2. `producer.yml` trains the model, tests the exporter, and uploads `producer/dist` as the `packaged-onnx-model` artifact.
3. `python-consumer.yml` downloads that artifact and runs the Python inference tests (including latency/consistency examples).
4. `java-consumer.yml` does the same for the Java inference tests (including latency/consistency examples).
5. `js-consumer.yml` does the same for the JavaScript inference tests (including latency/consistency examples).

The end result is one model package and three independent consumers proving that it can be used from Python, Java, and JavaScript (including browsers via WebAssembly).
