"""Explicit, audited decompositions of exporter fusions absent from ORT WebGPU."""
from __future__ import annotations

import onnx
from onnx import helper, TensorProto


def lower_rms_normalization(model):
    """Restore the native F32 multiply/mean/rsqrt/scale after ONNX fusion."""
    result, changes = [], []
    for node in model.graph.node:
        if node.op_type != "RMSNormalization" or node.domain:
            result.append(node)
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if len(node.input) != 2 or len(node.output) != 1 or attributes.get("stash_type", 1) != TensorProto.FLOAT:
            raise ValueError("unsupported RMSNormalization fusion contract: " + str(node))
        if attributes.get("axis", -1) != -1:
            raise ValueError("only native last-axis F32 RMS normalization is authorized")
        prefix = node.name + "/primitive/"
        axes, epsilon = prefix + "axes", prefix + "epsilon"
        model.graph.initializer.extend([
            helper.make_tensor(axes, TensorProto.INT64, [1], [-1]),
            helper.make_tensor(epsilon, TensorProto.FLOAT, [], [attributes.get("epsilon", 1e-5)])])
        operations = [("Mul", [node.input[0], node.input[0]], "square", {}),
                      ("ReduceMean", [prefix + "square", axes], "mean", {"keepdims": 1}),
                      ("Add", [prefix + "mean", epsilon], "variance", {}),
                      ("Sqrt", [prefix + "variance"], "root", {}),
                      ("Reciprocal", [prefix + "root"], "inverse", {}),
                      ("Mul", [node.input[0], prefix + "inverse"], "normalized", {}),
                      ("Mul", [node.input[1], prefix + "normalized"], "result", {})]
        for index, (op, inputs, suffix, attrs) in enumerate(operations):
            output = node.output[0] if index == len(operations) - 1 else prefix + suffix
            result.append(helper.make_node(op, inputs, [output], name=prefix + suffix, **attrs))
        changes.append({"node": node.name, "from": "RMSNormalization", "to": [o[0] for o in operations],
                        "reason": "No RMSNormalization shader in pinned ORT-WebGPU registration"})
    del model.graph.node[:]
    model.graph.node.extend(result)
    onnx.checker.check_model(model)
    return changes
