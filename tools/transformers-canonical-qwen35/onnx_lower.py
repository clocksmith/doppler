"""Explicit, audited decompositions of exporter fusions absent from ORT WebGPU."""
from __future__ import annotations

import onnx
from onnx import helper, TensorProto


def lower_rms_normalization(model, check=True):
    """Restore the native F32 multiply/mean/rsqrt/scale after ONNX fusion."""
    result, changes = [], []
    for node in model.graph.node:
        if node.op_type in ("LessOrEqual", "GreaterOrEqual") and not node.domain:
            strict = "Greater" if node.op_type == "LessOrEqual" else "Less"
            middle = node.name + "/strict-comparison"
            result.extend([helper.make_node(strict, list(node.input), [middle], name=middle),
                           helper.make_node("Not", [middle], list(node.output), name=node.name + "/not")])
            changes.append({"node": node.name, "from": node.op_type, "to": [strict, "Not"],
                            "reason": "Pinned ORT-WebGPU exposes strict comparison and Not operators"})
            continue
        if node.op_type == "Softplus" and not node.domain:
            if len(node.input) != 1 or len(node.output) != 1:
                raise ValueError("unsupported Softplus fusion contract")
            prefix = node.name + "/primitive/"
            zero, one = prefix + "zero", prefix + "one"
            model.graph.initializer.extend([helper.make_tensor(zero, TensorProto.FLOAT, [], [0.0]),
                                            helper.make_tensor(one, TensorProto.FLOAT, [], [1.0])])
            operations = [("Abs", [node.input[0]], "absolute"),
                          ("Neg", [prefix + "absolute"], "negative"),
                          ("Exp", [prefix + "negative"], "exponential"),
                          ("Add", [prefix + "exponential", one], "plus_one"),
                          ("Log", [prefix + "plus_one"], "logarithm"),
                          ("Greater", [node.input[0], zero], "positive"),
                          ("Where", [prefix + "positive", node.input[0], zero], "positive_part"),
                          ("Add", [prefix + "positive_part", prefix + "logarithm"], "result")]
            for index, (op, inputs, suffix) in enumerate(operations):
                output = node.output[0] if index == len(operations) - 1 else prefix + suffix
                result.append(helper.make_node(op, inputs, [output], name=prefix + suffix))
            changes.append({"node": node.name, "from": "Softplus", "to": [o[0] for o in operations],
                            "reason": "Stable F32 softplus decomposition; pinned ORT-WebGPU has no Softplus shader"})
            continue
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
    if check:
        onnx.checker.check_model(model)
    return changes
