"""Replay the first strict F32 state failure across hidden/norm/projection edges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from binding import digest, read_inputs, verify_bytes
from native_model import load_native_model, physical_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cpu-receipt", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy = json.loads(Path(__file__).with_name("contract.json").read_text())
    inputs = read_inputs(args.canonical, policy)
    verify_bytes(inputs, policy["hashChunkBytes"])
    artifact = json.loads((args.model / "receipt.json").read_text())
    cpu = json.loads(args.cpu_receipt.read_text())
    first = next(c for c in cpu["checks"] if not c["pass"])
    if (first["name"] != "present_conv.16" or first["case"] != "prefill-1-prefill-0" or
            artifact["contract"]["kvPhysicalDtype"] != "float32"):
        raise ValueError("diagnostic target differs from the recorded first F32 state failure")
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from export_text import cache_names, empty_cache

    graph_path = Path(artifact["sessionFiles"]["decoder_prefill"])
    graph = onnx.load(graph_path, load_external_data=False)
    nodes = {name: node for node in graph.graph.node for name in node.output}
    state_slice = nodes[first["name"]]
    concat = nodes[state_slice.input[0]]
    transposed = nodes[concat.input[1]]
    projection = nodes[transposed.input[0]]
    normalized = nodes[projection.input[0]]
    normalized_base = nodes[normalized.input[0]]
    hidden_name, inverse_name = normalized_base.input
    square_root = nodes[nodes[inverse_name].input[0]]
    variance_plus = nodes[square_root.input[0]]
    variance = nodes[variance_plus.input[0]]
    if [state_slice.op_type, concat.op_type, transposed.op_type, projection.op_type, normalized.op_type,
        normalized_base.op_type, nodes[inverse_name].op_type, square_root.op_type, variance_plus.op_type, variance.op_type] != [
            "Slice", "Concat", "Transpose", "MatMul", "Mul", "Mul", "Reciprocal", "Sqrt", "Add", "ReduceMean"]:
        raise ValueError("actual ONNX state-producer contract differs from the bounded diagnostic")
    extras = {"hidden": hidden_name, "normalized": projection.input[0], "projected": projection.output[0],
              "inverse": inverse_name, "variance": variance.output[0]}
    values = {v.name: v for v in list(graph.graph.value_info) + list(graph.graph.input) + list(graph.graph.output)}
    for name in extras.values():
        if name not in values:
            raise ValueError("missing exact internal shape/type metadata: " + name)
        graph.graph.output.append(values[name])
    diagnostic_graph = graph_path.with_name(graph_path.stem + ".state16-diagnostic.onnx")
    if diagnostic_graph.exists():
        raise ValueError("refusing to overwrite an existing diagnostic graph")
    onnx.save(graph, diagnostic_graph)
    onnx.checker.check_model(str(diagnostic_graph))
    torch.set_num_threads(policy["threads"])
    torch.set_num_interop_threads(policy["threads"])
    native, config, mappings = load_native_model(inputs, policy)
    captured = {}
    layer = native.model.layers[16]
    hooks = [layer.input_layernorm.register_forward_pre_hook(
                 lambda module, arguments: captured.update(hidden=arguments[0].detach().clone())),
             layer.input_layernorm.register_forward_hook(
                 lambda module, arguments, result: captured.update(normalized=result.detach().clone())),
             layer.linear_attn.in_proj_qkv.register_forward_hook(
                 lambda module, arguments, result: captured.update(projected=result.detach().clone()))]
    teacher = json.loads(args.teacher.read_text())
    code = next(p for p in teacher["prompts"] if p["id"] == "code")
    ids = torch.tensor([code["promptIds"][:1]], dtype=torch.int64)
    options = ort.SessionOptions()
    options.intra_op_num_threads = policy["threads"]
    options.inter_op_num_threads = policy["threads"]
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_mem_pattern = False
    session = ort.InferenceSession(str(diagnostic_graph), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        cache = tuple(t.to(torch.float32).numpy() for t in empty_cache(config))
        feeds = {"inputs_embeds": native.model.embed_tokens(ids).numpy(),
                 "position_ids": np.zeros((3, 1, 1), dtype=np.int64), "attention_mask": np.ones((1, 1), dtype=np.int64),
                 **{pair[0]: value for pair, value in zip(cache_names(config), cache)}}
        output = dict(zip([meta.name for meta in session.get_outputs()], session.run(None, feeds)))
        native(input_ids=ids, past_key_values=physical_cache(config, torch.float32), use_cache=True, logits_to_keep=1)
        for hook in hooks:
            hook.remove()
        observed = {key: torch.from_numpy(output[name]) for key, name in extras.items()}
        local_norm = layer.input_layernorm(observed["hidden"])
        local_projection = layer.linear_attn.in_proj_qkv(observed["normalized"])
        native_variance = observed["hidden"].pow(2).mean(-1, keepdim=True)
        inverse_from_observed_variance = torch.rsqrt(observed["variance"] + config.rms_norm_eps)
        reconstructed_state = torch.cat((torch.zeros(1, layer.linear_attn.conv_dim, config.linear_conv_kernel_dim),
                                          observed["projected"].transpose(1, 2)), -1)[:, :, -config.linear_conv_kernel_dim:]
        comparisons = []

        def compare(name, value, reference):
            difference = (value - reference).abs()
            comparisons.append({"name": name, "maxAbsoluteError": float(difference.max()),
                                "meanAbsoluteError": float(difference.mean()), "differentElements": int((difference != 0).sum()),
                                "elements": value.numel()})
            for engine, data in (("actual", value), ("reference", reference)):
                path = args.out / (name + "-" + engine + ".f32")
                path.write_bytes(data.contiguous().numpy().astype("<f4", copy=False).tobytes())

        compare("upstream-hidden", observed["hidden"], captured["hidden"])
        compare("total-normalized", observed["normalized"], captured["normalized"])
        compare("same-input-native-normalization", observed["normalized"], local_norm)
        compare("same-input-native-reduce-mean", observed["variance"], native_variance)
        compare("same-variance-native-rsqrt", observed["inverse"], inverse_from_observed_variance)
        compare("same-input-native-qkv-matmul", observed["projected"], local_projection)
        compare("total-projection", observed["projected"], captured["projected"])
        compare("state-binding-reconstruction", torch.from_numpy(output[first["name"]]), reconstructed_state)
    receipt = {"schema": "doppler.canonical-state16-diagnostic.v1", "inputs": inputs["inputs"], "parameterCount": len(mappings),
               "firstFailure": first, "comparisons": comparisons,
               "operatorChain": [{"name": node.name, "op": node.op_type, "input": list(node.input), "output": list(node.output)}
                                 for node in (state_slice, concat, transposed, projection, normalized, normalized_base,
                                              nodes[inverse_name], square_root, variance_plus, variance)],
               "sourceGraph": {"path": str(graph_path), "sha256": digest(graph_path)},
               "diagnosticGraph": {"path": str(diagnostic_graph), "sha256": digest(diagnostic_graph)},
               "cpuGate": {"path": str(args.cpu_receipt), "sha256": digest(args.cpu_receipt)},
               "artifacts": [{"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(args.out.glob("*.f32"))],
               "scope": "Same-input native F32 operator replay and exact state binding; original strict gate remains unchanged"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
