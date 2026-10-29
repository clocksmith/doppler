"""Make dynamic allocation geometry explicit inputs instead of I64 graph compute.

Only shape/index metadata derived from declared input dimensions is evaluated.
No learned tensor, activation, recurrent state or model-value arithmetic moves
to CPU. Runtime host recipes describe allocation geometry before graph dispatch.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper

from binding import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads((args.source / "receipt.json").read_text())
    if source.get("completed") is not True or source["contract"].get("integerInputs") != {
            "position_ids": "int32", "attention_mask": "int32"}:
        raise ValueError("host-shape lowering requires the validated bounded-index candidate")
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = deepcopy(source)
    receipt["sourceBoundedIndexArtifact"] = {"path": str(args.source / "receipt.json"), "sha256": digest(args.source / "receipt.json")}
    receipt["scope"] = "Explicit host allocation-geometry candidate; model arithmetic remains graph-only; no qualification"
    receipt["hostShapeLowerings"] = []
    binding = json.loads(Path(source["initializerBinding"]["path"]).read_text())
    for graph in receipt["graphs"]:
        phase, path = graph["mode"], Path(graph["path"])
        model = onnx.load(path, load_external_data=False)
        producers = {name: node for node in model.graph.node for name in node.output}
        values = {value.name: value for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)}
        types = {name: value.type.tensor_type.elem_type for name, value in values.items()}
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        initializer_bytes = {tensor.name: tensor.SerializeToString() for tensor in model.graph.initializer}
        types.update({name: tensor.data_type for name, tensor in initializers.items()})
        public_inputs = {value.name: value for value in model.graph.input}
        old_controls = source["contract"]["hostGeometryInputs"][phase]
        controls_by_name = {control["name"]: control for control in old_controls}
        memo = {}

        def constant(name):
            tensor = initializers[name]
            if tensor.data_type != T.INT64 or np.prod(tensor.dims, dtype=np.int64) > 64:
                raise ValueError("allocation metadata must use bounded I64 constants")
            return numpy_helper.to_array(tensor, base_dir=str(path.parent))

        def expressions(name):
            if name in memo:
                return memo[name]
            if name in initializers:
                array = constant(name)
                result = np.empty(array.shape, dtype=object)
                for index in np.ndindex(array.shape):
                    result[index] = {"constant": int(array[index])}
            elif name in controls_by_name:
                control = controls_by_name[name]
                recipes = control["values"] if "values" in control else [{key: control[key] for key in
                    ("sourceTensor", "axis", "multiplier", "addend")}]
                result = np.empty(len(recipes), dtype=object)
                result[:] = recipes
                result = result.reshape(control["shape"])
            else:
                node = producers[name]
                attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
                if node.op_type == "Shape":
                    origin = public_inputs[node.input[0]]
                    rank = len(origin.type.tensor_type.shape.dim)
                    axes = list(range(rank))[slice(attrs.get("start", 0), attrs.get("end", rank))]
                    result = np.empty(len(axes), dtype=object)
                    result[:] = [{"sourceTensor": origin.name, "axis": axis, "multiplier": 1, "addend": 0} for axis in axes]
                elif node.op_type == "Squeeze":
                    axes = tuple(int(x) for x in constant(node.input[1]).reshape(-1)) if len(node.input) > 1 else None
                    result = np.squeeze(expressions(node.input[0]), axis=axes)
                elif node.op_type == "Unsqueeze":
                    axes = tuple(int(x) for x in constant(node.input[1]).reshape(-1))
                    result = np.expand_dims(expressions(node.input[0]), axis=axes)
                elif node.op_type == "Gather":
                    result = np.asarray(np.take(expressions(node.input[0]), constant(node.input[1]), axis=attrs.get("axis", 0)), dtype=object)
                elif node.op_type == "Concat":
                    result = np.concatenate([expressions(value) for value in node.input], axis=attrs["axis"])
                elif node.op_type == "Reshape":
                    result = expressions(node.input[0]).reshape(tuple(int(x) for x in constant(node.input[1])))
                else:
                    raise ValueError("unrecognized allocation-geometry producer: " + node.op_type + " " + node.name)
            result = np.asarray(result, dtype=object)
            if result.size > 64:
                raise ValueError("allocation recipe unexpectedly contains model-sized data")
            memo[name] = result
            return result

        generated, replacements = [], {}
        for node in model.graph.node:
            if all(types.get(output) == T.INT64 for output in node.output):
                continue
            for index, name in enumerate(node.input):
                if types.get(name) != T.INT64 or name in initializers or name in public_inputs:
                    continue
                if name not in replacements:
                    recipe = expressions(name)
                    target = "host_shape_" + str(len(generated))
                    model.graph.input.append(helper.make_tensor_value_info(target, T.INT64, list(recipe.shape)))
                    generated.append({"name": target, "type": "int64", "shape": list(recipe.shape),
                                      "values": recipe.reshape(-1).tolist(), "replaces": name})
                    replacements[name] = target
                node.input[index] = replacements[name]
        needed = {value.name for value in model.graph.output}
        kept = []
        for node in reversed(model.graph.node):
            if any(name in needed for name in node.output):
                kept.append(node)
                needed.update(node.input)
        del model.graph.node[:]
        model.graph.node.extend(reversed(kept))
        controls = [control for control in old_controls + generated if control["name"] in needed]
        control_names = {control["name"] for control in old_controls + generated}
        kept_inputs = [value for value in model.graph.input if value.name not in control_names or value.name in needed]
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)
        for parameter in binding["parameters"]:
            for target in parameter["bindings"]:
                if target["graph"] == "decoder_" + phase and target["initializer"] not in needed:
                    raise ValueError("canonical initializer became unreachable")
        for node in model.graph.node:
            if any(types.get(output) == T.INT64 for output in node.output):
                raise ValueError("I64 allocation computation remains in the GPU graph: " + node.name)
        if any(tensor.SerializeToString() != initializer_bytes[tensor.name] for tensor in model.graph.initializer):
            raise ValueError("allocation lowering changed an initializer protobuf")
        target = path.with_name("decoder_" + phase + "." + args.out.name + ".onnx")
        if target.exists():
            raise ValueError("refusing to overwrite a host-shape candidate")
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        graph["path"], graph["sha256"], graph["inputs"] = str(target), digest(target), [value.name for value in model.graph.input]
        receipt["sessionFiles"]["decoder_" + phase] = str(target)
        receipt["contract"]["hostGeometryInputs"][phase] = controls
        receipt["hostShapeLowerings"].append({"phase": phase, "geometryInputs": controls,
            "canonicalInitializerProtobufsUnchanged": True, "modelValueArithmeticOnCpu": False})
    shutil.copytree(args.source / "sources", args.out / "sources")
    shutil.copyfile(Path(__file__), args.out / "sources" / "variant_host_shapes.py")
    receipt["sourceFiles"]["variant_host_shapes.py"] = digest(Path(__file__))
    for filename in ("config.json", "tokenizer.json", "initializer-binding.json"):
        shutil.copyfile(args.source / filename, args.out / filename)
    receipt["initializerBinding"]["path"] = str(args.out / "initializer-binding.json")
    files = [Path(entry["path"]) for entry in source["artifacts"] if entry["path"].endswith(".data")]
    files.extend(Path(value) for value in receipt["sessionFiles"].values())
    receipt["artifacts"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(files)]
    (args.out / "lane-policy.json").write_text(json.dumps(receipt["contract"], indent=2) + "\n")
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
