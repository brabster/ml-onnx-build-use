from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


MODEL_FILE_NAME = "setosa_probability.onnx"
METADATA_FILE_NAME = "setosa_probability_metadata.json"


def train_and_package(output_dir: Path) -> Path:
    iris = load_iris()
    features = iris.data.astype(np.float32)
    # Binary labels: 1 = Iris Setosa, 0 = not Setosa
    labels = (iris.target == 0).astype(int)

    classifier = LogisticRegression(max_iter=400, random_state=0)
    classifier.fit(features, labels)

    onnx_model = to_onnx(
        classifier,
        initial_types=[("features", FloatTensorType([None, features.shape[1]]))],
        options={id(classifier): {"zipmap": False}},
        target_opset=18,
    )

    # The binary classifier outputs [label, probabilities], where probabilities has
    # shape [n, 2].  Post-process the graph to output only the probability of class 1
    # (Setosa) as a single float32 value.
    graph = onnx_model.graph
    graph.node.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["_class_index"],
            value=numpy_helper.from_array(np.array(1, dtype=np.int64)),
        )
    )
    graph.node.append(
        helper.make_node(
            "Gather",
            inputs=["probabilities", "_class_index"],
            outputs=["setosa_probability"],
            axis=1,
        )
    )
    while len(graph.output) > 0:
        graph.output.pop()
    graph.output.append(
        helper.make_tensor_value_info("setosa_probability", TensorProto.FLOAT, [None])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILE_NAME
    metadata_path = output_dir / METADATA_FILE_NAME

    onnx.save_model(onnx_model, model_path)

    sample_input = features[0].tolist()
    metadata = {
        "contract_version": 1,
        "dataset": "iris",
        "description": "Probability that an iris flower is Iris Setosa",
        "input_name": "features",
        "feature_count": features.shape[1],
        "output_name": "setosa_probability",
        "output_kind": "probability",
        "prediction_tolerance": 0.001,
        "feature_names": iris.feature_names,
        "sample_input": sample_input,
        "expected_sample_prediction": float(
            classifier.predict_proba([sample_input])[0, 1]
        ),
        "model_file": MODEL_FILE_NAME,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the setosa probability model and export it as ONNX."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dist",
        help="Directory that will receive the packaged ONNX model.",
    )
    arguments = parser.parse_args()
    model_path = train_and_package(arguments.output_dir)
    print(f"Packaged model written to {model_path}")


if __name__ == "__main__":
    main()
