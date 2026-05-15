# ml-onnx-build-use

This repository is a small, plain-English example of one ONNX model being produced once and consumed in two different runtimes.

## What is in the repo?

- `producer/` trains two models with scikit-learn and exports them as ONNX files.
  - `train_and_package.py` — iris flower classifier that outputs a **long** class label (0, 1, or 2).
  - `train_and_package_probability.py` — binary classifier that outputs a **float** probability of a flower being Iris Setosa (a value between 0 and 1).
- `python_consumer/` loads both ONNX files with ONNX Runtime for Python and checks inferences in tests.
- `java_consumer/` loads the same ONNX files with ONNX Runtime for Java and checks the same inferences in tests.
- `.github/workflows/` contains three reusable pipelines plus one orchestration workflow. The producer pipeline publishes the packaged models as a GitHub Actions artifact. The two consumer pipelines download that artifact and run their tests against it.

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

## GitHub Actions flow

1. `ci.yml` runs on pushes and pull requests, then calls the three reusable workflows in order.
2. `producer.yml` trains the model, tests the exporter, and uploads `producer/dist` as the `packaged-onnx-model` artifact.
3. `python-consumer.yml` downloads that artifact and runs the Python inference test.
4. `java-consumer.yml` does the same for the Java inference test.

The end result is one model package and two independent consumers proving that it can be used from both ecosystems.

## The model contract

The producer and consumer need to agree on the model's interface — its **inputs**, **outputs**, and **semantics**.  Once that interface is stable, both sides can evolve independently without co-ordinated changes.

### What is in the contract?

| Element | Value |
|---------|-------|
| Input tensor name | `features` |
| Input tensor type | `float32` |
| Input tensor shape | `[1, 4]` — one row of four measurements (sepal length, sepal width, petal length, petal width in cm) |
| `iris_classifier` output | A single `int64` class label: `0` = Setosa, `1` = Versicolor, `2` = Virginica |
| `setosa_probability` output | A single `float32` probability that the flower is Iris Setosa (a value between 0 and 1) |

### How is the contract tested?

The producer publishes a **metadata file** alongside each ONNX file.  The metadata file includes a `sample_input` and the matching `expected_sample_prediction`, giving every consumer a concrete example to assert against.

```
producer/dist/
  iris_classifier.onnx
  model_metadata.json          ← sample_input + expected_sample_prediction
  setosa_probability.onnx
  setosa_probability_metadata.json
```

Both consumers read these metadata files to drive their tests.  A consumer never hardcodes the test values — it reads them from the metadata.  If the producer retrains the model or updates the sample, the consumers automatically pick up the new values; no co-ordinated change is required.

The only stable part of the contract is the **metadata file names** (`model_metadata.json` and `setosa_probability_metadata.json`) and the **field names** (`sample_input`, `expected_sample_prediction`).  As long as those are kept consistent, producer and consumer are fully decoupled.
