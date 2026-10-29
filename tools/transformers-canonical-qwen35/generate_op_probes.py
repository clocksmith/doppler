"""Create tiny exact-dtype GPU admission probes; no model weights or fallback."""
import argparse
import json
from pathlib import Path

import onnx
from onnx import helper, TensorProto


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
cases = []


def add(name, op, tensors, output_type, output_shape, expected=None, attributes=None):
    inputs = [helper.make_tensor_value_info(t["name"], t["onnxType"], t.get("metadataShape", t["shape"])) for t in tensors]
    output = helper.make_tensor_value_info("output", output_type, output_shape)
    graph = helper.make_graph([helper.make_node(op, [t["name"] for t in tensors], ["output"], name=name, **(attributes or {}))],
                              name, inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, args.out / (name + ".onnx"))
    cases.append({"name": name, "file": name + ".onnx", "inputs": tensors, "expected": expected})


for dtype, kind in (("int64", TensorProto.INT64), ("int32", TensorProto.INT32)):
    add("range-" + dtype, "Range", [{"name": name, "type": dtype, "onnxType": kind, "shape": [], "values": [value]}
        for name, value in (("start", 0), ("limit", 3), ("delta", 1))], kind, [3], [0, 1, 2])
    add("greater-" + dtype, "Greater", [{"name": "value", "type": dtype, "onnxType": kind, "shape": [1, 3], "values": [1, 0, 1]},
        {"name": "zero", "type": dtype, "onnxType": kind, "shape": [], "values": [0]}], TensorProto.BOOL, [1, 3], [1, 0, 1])
add("cast-int64-f32", "Cast", [{"name": "value", "type": "int64", "onnxType": TensorProto.INT64,
    "shape": [3, 1, 2], "values": [0, 1, 0, 1, 0, 1]}], TensorProto.FLOAT, [3, 1, 2], [0, 1, 0, 1, 0, 1], {"to": TensorProto.FLOAT})
add("shape-dynamic", "Shape", [{"name": "value", "type": "float32", "onnxType": TensorProto.FLOAT,
    "shape": [1, 3, 4], "metadataShape": [1, "sequence_length", 4]}], TensorProto.INT64, [3], [1, 3, 4])
for dtype, kind in (("int64", TensorProto.INT64), ("float32", TensorProto.FLOAT)):
    add("pow-square-" + dtype, "Pow", [{"name": "value", "type": "float32", "onnxType": TensorProto.FLOAT,
        "shape": [1, 2], "values": [2, 4]}, {"name": "exponent", "type": dtype, "onnxType": kind, "shape": [], "values": [2]}],
        TensorProto.FLOAT, [1, 2], [4, 16])
add("matmul-rank5", "MatMul", [{"name": "left", "type": "float32", "onnxType": TensorProto.FLOAT, "shape": [1, 2, 2, 3, 4]},
    {"name": "right", "type": "float32", "onnxType": TensorProto.FLOAT, "shape": [1, 2, 2, 4, 3]}],
    TensorProto.FLOAT, [1, 2, 2, 3, 3], [0] * 36)
for dtype, kind in (("int32", TensorProto.INT32), ("int64", TensorProto.INT64)):
    add("shape-" + dtype, "Shape", [{"name": "value", "type": dtype, "onnxType": kind,
        "shape": [1, 3, 4], "metadataShape": [1, "sequence_length", 4]}], TensorProto.INT64, [3], [1, 3, 4])
add("cast-int32-f32", "Cast", [{"name": "value", "type": "int32", "onnxType": TensorProto.INT32,
    "shape": [3, 1, 2], "values": [0, 1, 0, 1, 0, 1]}], TensorProto.FLOAT, [3, 1, 2], [0, 1, 0, 1, 0, 1], {"to": TensorProto.FLOAT})
add("concat-int64", "Concat", [{"name": "left", "type": "int64", "onnxType": TensorProto.INT64, "shape": [1], "values": [1]},
    {"name": "right", "type": "int64", "onnxType": TensorProto.INT64, "shape": [2], "values": [3, 4]}],
    TensorProto.INT64, [3], [1, 3, 4], {"axis": 0})
add("gather-int64", "Gather", [{"name": "data", "type": "int64", "onnxType": TensorProto.INT64, "shape": [3], "values": [1, 3, 4]},
    {"name": "index", "type": "int64", "onnxType": TensorProto.INT64, "shape": [], "values": [1]}],
    TensorProto.INT64, [], [3], {"axis": 0})
add("expand-f32", "Expand", [{"name": "data", "type": "float32", "onnxType": TensorProto.FLOAT,
    "shape": [1, 1, 4], "values": [1, 2, 3, 4]},
    {"name": "shape", "type": "int64", "onnxType": TensorProto.INT64, "shape": [3], "values": [1, 3, 4]}],
    TensorProto.FLOAT, [1, 3, 4], [1, 2, 3, 4] * 3)
composed = {"name": "concat-reshape-metadata", "file": "concat-reshape-metadata.onnx", "inputs": [
    {"name": "data", "type": "float32", "onnxType": TensorProto.FLOAT, "shape": [1, 3, 4], "values": list(range(12))},
    {"name": "left", "type": "int64", "onnxType": TensorProto.INT64, "shape": [1], "values": [1]},
    {"name": "right", "type": "int64", "onnxType": TensorProto.INT64, "shape": [2], "values": [3, 4]}], "expected": list(range(12))}
graph = helper.make_graph([helper.make_node("Concat", ["left", "right"], ["target"], axis=0),
                          helper.make_node("Reshape", ["data", "target"], ["output"])], composed["name"],
    [helper.make_tensor_value_info(t["name"], t["onnxType"], t["shape"]) for t in composed["inputs"]],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)], ir_version=10)
onnx.checker.check_model(model)
onnx.save(model, args.out / composed["file"])
cases.append(composed)
add("reshape-direct-metadata", "Reshape", [{"name": "data", "type": "float32", "onnxType": TensorProto.FLOAT,
    "shape": [1, 3, 4], "values": list(range(12))}, {"name": "shape", "type": "int64", "onnxType": TensorProto.INT64,
    "shape": [3], "values": [1, 3, 4]}], TensorProto.FLOAT, [1, 3, 4], list(range(12)))
add("gather-f32-rank5", "Gather", [{"name": "data", "type": "float32", "onnxType": TensorProto.FLOAT,
    "shape": [1, 2, 2, 3, 4]}, {"name": "index", "type": "int64", "onnxType": TensorProto.INT64,
    "shape": [], "values": [1]}], TensorProto.FLOAT, [1, 2, 2, 4], [0] * 16, {"axis": 3})
for operator in ("Exp", "Neg"):
    name = "constant-" + operator.lower() + "-then-mul"
    tensors = [{"name": "data", "type": "float32", "onnxType": TensorProto.FLOAT,
                "shape": [16], "values": list(range(1, 17))}]
    constant_value = [0.0] * 16 if operator == "Exp" else [-1.0] * 16
    initializer = helper.make_tensor("coefficient", TensorProto.FLOAT, [16], constant_value)
    graph = helper.make_graph([helper.make_node(operator, ["coefficient"], ["prepared"]),
                               helper.make_node("Mul", ["data", "prepared"], ["output"])], name,
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [16])], [initializer])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, args.out / (name + ".onnx"))
    cases.append({"name": name, "file": name + ".onnx", "inputs": tensors, "expected": list(range(1, 17))})
(args.out / "probes.json").write_text(json.dumps(cases, indent=2) + "\n")
