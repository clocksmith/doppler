"""Lower bounded position/mask integers and declare host geometry control ports.

No runtime tensor values are computed on CPU. Host ports contain only declared
tensor dimensions and model head geometry, analogous to allocation descriptors.
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
    config = json.loads((args.source / "config.json").read_text())
    if source.get("completed") is not True or source["contract"]["kvPhysicalDtype"] != "float32":
        raise ValueError("bounded-index variant requires completed distinct F32-KV construction")
    if source["contract"]["maxSequenceLength"] > np.iinfo(np.int32).max:
        raise ValueError("declared index bound does not fit I32")
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = deepcopy(source)
    receipt["sourceF32Artifact"] = {"path": str(args.source / "receipt.json"), "sha256": digest(args.source / "receipt.json")}
    receipt["scope"] = "Bounded-index GPU-admission candidate; no numerical, GPU, TJS or speed qualification"
    receipt["contract"]["integerInputs"] = {"position_ids": "int32", "attention_mask": "int32"}
    receipt["contract"]["hostGeometryInputs"] = {}
    receipt["indexLowerings"] = []
    learned = json.loads(Path(source["initializerBinding"]["path"]).read_text())
    for graph in receipt["graphs"]:
        phase = graph["mode"]
        path = Path(graph["path"])
        model = onnx.load(path, load_external_data=False)
        initializers = {t.name: t for t in model.graph.initializer}
        original_parameters = {t.name: t.SerializeToString() for t in model.graph.initializer}
        metadata = {v.name: v for v in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)}
        types = {name: value.type.tensor_type.elem_type for name, value in metadata.items()}
        types.update({name: value.data_type for name, value in initializers.items()})
        replacements, controls, changes, remove = {}, [], [], set()

        def value(name):
            tensor = initializers[name]
            if np.prod(tensor.dims, dtype=np.int64) > 64:
                raise ValueError("geometry constant unexpectedly large")
            return numpy_helper.to_array(tensor, base_dir=str(path.parent))

        def host(name, old, dtype, source_tensor, axis, shape, multiplier=1, addend=0):
            model.graph.input.append(helper.make_tensor_value_info(name, dtype, shape))
            types[name] = dtype
            if old:
                replacements[old] = name
            controls.append({"name": name, "type": "int32" if dtype == T.INT32 else "int64", "shape": shape,
                "sourceTensor": source_tensor, "axis": axis, "multiplier": multiplier, "addend": addend})
            return name

        shapes = [n for n in model.graph.node if n.op_type == "Shape"]
        ranges = [n for n in model.graph.node if n.op_type == "Range"]
        if len(shapes) != 1 or len(ranges) != 1:
            raise ValueError("bounded canonical graph must have one shared dynamic shape and range")
        shape = shapes[0]
        if phase == "prefill":
            if shape.input[0] != "attention_mask":
                raise ValueError("prefill geometry owner changed")
            integer_multiplications = [n for n in model.graph.node if n.op_type == "Mul" and types.get(n.output[0]) == T.INT64]
            if len(integer_multiplications) != 1:
                raise ValueError("prefill flattened-head shape arithmetic changed")
            multiply = integer_multiplications[0]
            constant = next(name for name in multiply.input if name in initializers)
            heads = int(value(constant))
            if heads != config["text_config"]["linear_num_value_heads"]:
                raise ValueError("flattened geometry does not match declared value heads")
            host("host_token_heads", multiply.output[0], T.INT64, "attention_mask", 1, [], multiplier=heads)
            remove.add(multiply.name)
        else:
            additions = [n for n in model.graph.node if n.op_type == "Add" and types.get(n.output[0]) == T.INT64]
            if len(additions) != 2:
                raise ValueError("decode geometry arithmetic changed")
            for add in additions:
                shape_dims = [d.dim_value for d in metadata[add.output[0]].type.tensor_type.shape.dim]
                if shape_dims == []:
                    host("host_total_length_i64", add.output[0], T.INT64, "attention_mask", 1, [])
                elif shape_dims == [1]:
                    host("host_query_position_i32", add.output[0], T.INT32, shape.input[0], 2, [1])
                else:
                    raise ValueError("unexpected decode scalar geometry shape")
                remove.add(add.name)
        range_limit = host("host_range_limit_i32", None, T.INT32, "attention_mask", 1, [])
        for input_value in model.graph.input:
            if input_value.name in ("position_ids", "attention_mask"):
                if input_value.type.tensor_type.elem_type != T.INT64:
                    raise ValueError("source application index input is not I64")
                input_value.type.tensor_type.elem_type = T.INT32
                types[input_value.name] = T.INT32

        cloned = {}

        def i32_constant(name):
            if name in cloned:
                return cloned[name]
            tensor = initializers.get(name)
            if tensor is None or tensor.data_type != T.INT64:
                raise ValueError("nonconstant I64 arithmetic still reaches a GPU numeric node: " + name)
            array = value(name)
            if not ((array >= np.iinfo(np.int32).min).all() and (array <= np.iinfo(np.int32).max).all()):
                raise ValueError("integer constant overflows declared I32 lowering")
            target = name + "/bounded_i32"
            model.graph.initializer.append(numpy_helper.from_array(array.astype(np.int32), target))
            types[target] = T.INT32
            cloned[name] = target
            return target

        nodes = []
        for node in model.graph.node:
            if node.name in remove:
                continue
            for index, name in enumerate(node.input):
                node.input[index] = replacements.get(name, name)
            if node.op_type == "Pow":
                if node.input[1] not in initializers or not bool(np.all(value(node.input[1]) == 2)):
                    raise ValueError("only exact native square powers may lower to multiplication")
                node.op_type = "Mul"
                node.input[1] = node.input[0]
                changes.append({"node": node.name, "from": "Pow(x,2)", "to": "Mul(x,x)"})
            if node.op_type == "Range":
                node.input[0], node.input[1], node.input[2] = i32_constant(node.input[0]), range_limit, i32_constant(node.input[2])
                types[node.output[0]] = T.INT32
            if node.op_type in ("Add", "Sub", "Mul", "Div", "Greater", "Less", "Equal") and any(
                    types.get(name) == T.INT32 for name in node.input):
                for index, name in enumerate(node.input):
                    if types.get(name) == T.INT64:
                        node.input[index] = i32_constant(name)
                if node.op_type not in ("Greater", "Less", "Equal"):
                    types[node.output[0]] = T.INT32
            if node.op_type in ("Slice", "Gather", "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Expand", "Identity"):
                types[node.output[0]] = types[node.input[0]]
            nodes.append(node)
        needed = {v.name for v in model.graph.output}
        kept = []
        for node in reversed(nodes):
            if any(name in needed for name in node.output):
                kept.append(node)
                needed.update(node.input)
        del model.graph.node[:]
        model.graph.node.extend(reversed(kept))
        for parameter in learned["parameters"]:
            for binding in parameter["bindings"]:
                if binding["graph"] == "decoder_" + phase:
                    name = binding["initializer"]
                    if name not in needed or initializers[name].SerializeToString() != original_parameters[name]:
                        raise ValueError("learned initializer identity/reachability changed")
        for annotation in model.graph.value_info:
            if annotation.name in types:
                annotation.type.tensor_type.elem_type = types[annotation.name]
        model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=False)
        target = path.with_name("decoder_" + phase + "." + args.out.name + ".onnx")
        if target.exists():
            raise ValueError("refusing to overwrite bounded-index graph")
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        graph["path"], graph["sha256"] = str(target), digest(target)
        graph["inputs"] = [v.name for v in model.graph.input]
        receipt["sessionFiles"]["decoder_" + phase] = str(target)
        receipt["contract"]["hostGeometryInputs"][phase] = controls
        receipt["indexLowerings"].append({"phase": phase, "exactSquares": changes, "controls": controls,
            "canonicalInitializersByteUnchanged": True, "indexRange": [0, source["contract"]["maxSequenceLength"]]})
    receipt["sourceFiles"]["variant_gpu_indices.py"] = digest(Path(__file__))
    shutil.copytree(args.source / "sources", args.out / "sources")
    shutil.copyfile(Path(__file__), args.out / "sources" / "variant_gpu_indices.py")
    for filename in ("config.json", "tokenizer.json", "initializer-binding.json"):
        shutil.copyfile(args.source / filename, args.out / filename)
    receipt["initializerBinding"]["path"] = str(args.out / "initializer-binding.json")
    files = [Path(a["path"]) for a in source["artifacts"] if a["path"].endswith(".data")]
    files.extend(Path(value) for value in receipt["sessionFiles"].values())
    receipt["artifacts"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(files)]
    (args.out / "lane-policy.json").write_text(json.dumps(receipt["contract"], indent=2) + "\n")
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
