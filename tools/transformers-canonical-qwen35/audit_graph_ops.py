"""Extract exact-type/rank node signatures for no-fallback GPU admission only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from binding import digest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--graph", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
model = onnx.load(args.graph, load_external_data=False)
metadata = {v.name: v for v in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)}
initializers = {t.name: t for t in model.graph.initializer}
for tensor in model.graph.initializer:
    metadata[tensor.name] = helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims))
groups = {}
for node in model.graph.node:
    signature = {"op": node.op_type, "domain": node.domain,
                 "attributes": [a.SerializeToString().hex() for a in node.attribute], "inputs": [], "outputs": []}
    for role, names in (("inputs", node.input), ("outputs", node.output)):
        for name in names:
            value = metadata[name]
            shape = [d.dim_value if d.HasField("dim_value") else "?" for d in value.type.tensor_type.shape.dim]
            entry = {"type": value.type.tensor_type.elem_type, "shape": shape}
            if name in initializers and initializers[name].data_type == onnx.TensorProto.INT64:
                array = numpy_helper.to_array(initializers[name], base_dir=str(args.graph.parent))
                entry["integerConstant"] = array.tolist()
            signature[role].append(entry)
    key = json.dumps(signature, sort_keys=True)
    if key in groups:
        groups[key]["sourceNodes"].append(node.name)
        continue
    index = len(groups)
    inputs, constants, input_descriptors = [], [], []
    for name in dict.fromkeys(node.input):
        if not name:
            continue
        tensor = initializers.get(name)
        if tensor is not None and np.prod(tensor.dims, dtype=np.int64) <= 262144:
            constants.append(numpy_helper.from_array(numpy_helper.to_array(tensor, base_dir=str(args.graph.parent)), name))
        else:
            value = deepcopy(metadata[name])
            inputs.append(value)
            input_descriptors.append({"name": name, "onnxType": value.type.tensor_type.elem_type,
                                      "shape": [d.dim_value if d.HasField("dim_value") else "dynamic" for d in value.type.tensor_type.shape.dim]})
    graph = helper.make_graph([deepcopy(node)], "signature_" + str(index), inputs,
                              [deepcopy(metadata[name]) for name in node.output], constants)
    probe = helper.make_model(graph, opset_imports=model.opset_import, ir_version=model.ir_version)
    onnx.checker.check_model(probe)
    filename = f"signature-{index}.onnx"
    onnx.save(probe, args.out / filename)
    groups[key] = {"name": f"{index}-{node.op_type}", "file": filename, "inputs": input_descriptors,
                   "expected": None, "sourceNodes": [node.name], "signature": signature}
(args.out / "probes.json").write_text(json.dumps(list(groups.values()), indent=2) + "\n")
(args.out / "source.json").write_text(json.dumps({"graph": str(args.graph), "sha256": digest(args.graph),
    "scope": "Node capability only; large initializer values are graph inputs, no model execution", "signatures": len(groups)}, indent=2))
print(json.dumps({"signatures": len(groups)}), flush=True)
