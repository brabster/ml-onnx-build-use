# ml-onnx-build-use

This repository is a small, plain-English example of one ONNX model being produced once and consumed in two different runtimes.

## What is in the repo?

- `producer/` trains a simple iris flower classifier with scikit-learn and exports it as `iris_classifier.onnx`.
- `python_consumer/` loads that ONNX file with ONNX Runtime for Python and checks an inference in a test.
- `java_consumer/` loads the same ONNX file with ONNX Runtime for Java and checks the same inference in a test.
- `.github/workflows/` contains three pipelines. The producer pipeline publishes the packaged model as a GitHub Actions artifact. The two consumer pipelines download that artifact and run their tests against it.

## Why the iris dataset?

The iris dataset ships with scikit-learn, is public, and is small enough to keep the example focused on packaging and inference instead of data wrangling.

## Run the example locally

1. Train and package the model.

   ```bash
   python3 -m pip install -r producer/requirements.txt
   python3 producer/train_and_package.py --output-dir producer/dist
   ```

2. Run the Python consumer test.

   ```bash
   python3 -m pip install -r python_consumer/requirements.txt
   MODEL_PATH=producer/dist/iris_classifier.onnx python3 -m pytest python_consumer/tests
   ```

3. Run the Java consumer test.

   ```bash
   cd java_consumer
   MODEL_PATH=../producer/dist/iris_classifier.onnx mvn --batch-mode test
   ```

## GitHub Actions flow

1. `producer.yml` trains the model, tests the exporter, and uploads `producer/dist` as the `packaged-onnx-model` artifact.
2. `python-consumer.yml` starts when the producer workflow succeeds, downloads the artifact, and runs the Python inference test.
3. `java-consumer.yml` does the same for the Java inference test.

The end result is one model package and two independent consumers proving that it can be used from both ecosystems.
