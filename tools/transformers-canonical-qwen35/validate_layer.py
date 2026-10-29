"""Compare exporter lowering against unchanged native HF using canonical weights."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback

from binding import digest, read_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    lane = policy["linearExport"]
    inputs = read_inputs(args.canonical, policy)
    expected = policy["frameworks"] | policy["exportFrontend"]["frameworks"]
    versions = {name: importlib.metadata.version(name) for name in expected}
    if expected != versions:
        raise ValueError("pinned framework mismatch: " + repr(versions))
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from export_layer import LinearAttentionExport
    from onnx_lower import lower_rms_normalization

    if hf.is_fast_path_available or hf.FusedRMSNormGated is not None:
        raise ValueError("validation requires unchanged CPU native HF functions")
    torch.set_num_threads(policy["threads"])
    torch.manual_seed(lane["seed"])
    config = Qwen3_5TextConfig(**inputs["config"]["text_config"])
    native = hf.Qwen3_5GatedDeltaNet(config, lane["layer"]).eval()
    prefix = f"{policy['sourcePrefix']}layers.{lane['layer']}.linear_attn."
    declared = {t["name"]: t for t in inputs["manifest"]["tensors"] if t["name"].startswith(prefix)}
    state = {}
    with safe_open(str(inputs["weights"]), framework="pt", device="cpu") as source:
        for key, expected_tensor in native.state_dict().items():
            name = prefix + key
            value = source.get_tensor(name)
            if list(value.shape) != declared[name]["shape"] or value.dtype != torch.float32:
                raise ValueError("canonical layer shape or dtype mismatch: " + name)
            raw = value.numpy().tobytes()
            if len(raw) != declared[name]["bytes"]:
                raise ValueError("canonical layer byte count mismatch: " + name)
            if hashlib.sha256(raw).hexdigest() != declared[name]["canonicalBytesSha256"]:
                raise ValueError("canonical layer byte digest mismatch: " + name)
            state[key] = value
    if set(state) != {name[len(prefix):] for name in declared}:
        raise ValueError("canonical layer has unused parameters")
    native.load_state_dict(state, strict=True)
    adapters = {mode: LinearAttentionExport(native, lane["prefillBound"], mode).eval()
                for mode in ("prefill", "decode")}

    class CacheLayer:
        pass

    class NativeCache:
        def __init__(self, previous, conv, recurrent):
            self.previous = previous
            self.layers = [CacheLayer() for _ in range(lane["layer"] + 1)]
            self.layers[lane["layer"]].conv_states = conv.clone()
            self.layers[lane["layer"]].recurrent_states = recurrent.clone()

        def has_previous_state(self, index):
            return self.previous

        def update_conv_state(self, value, index):
            self.layers[index].conv_states = value
            return value

        def update_recurrent_state(self, value, index):
            self.layers[index].recurrent_states = value
            return value

    conv_shape = (1, native.conv_dim, native.conv_kernel_size)
    recurrent_shape = (1, native.num_v_heads, native.head_k_dim, native.head_v_dim)
    receipt = {"schema": "doppler.transformers-qwen35-custom-linear.v1", "inputs": inputs["inputs"],
               "frameworks": versions, "policySha256": digest(policy_path),
               "sourceSha256": {"native": digest(Path(hf.__file__)),
                                 "adapter": digest(Path(__file__).with_name("export_layer.py"))},
               "parameters": [{"name": name, "sha256": t["canonicalBytesSha256"]} for name, t in declared.items()],
               "scope": "Canonical layer CPU parity and ONNX structure only; no GPU execution or full model proof",
               "checks": [], "exports": []}
    onnx_cases = []

    def compare(label, actual, expected_result):
        for name, value, reference in zip(("output", "conv", "recurrent"), actual, expected_result):
            difference = (value - reference).abs()
            limit = lane["absoluteTolerance"] + lane["relativeTolerance"] * reference.abs()
            passed = bool(torch.isfinite(value).all() and torch.isfinite(reference).all() and (difference <= limit).all())
            check = {"case": label, "tensor": name, "shape": list(value.shape), "pass": passed,
                     "maxAbsoluteError": float(difference.max()), "meanAbsoluteError": float(difference.mean()),
                     "maxReferenceMagnitude": float(reference.abs().max())}
            receipt["checks"].append(check)
            if not passed:
                raise ValueError("native HF comparison failed: " + json.dumps(check))

    def native_call(hidden, previous, conv, recurrent):
        cache = NativeCache(previous, conv, recurrent)
        output = native(hidden, cache_params=cache)
        return output, cache.layers[lane["layer"]].conv_states, cache.layers[lane["layer"]].recurrent_states

    def check_case(label, mode, hidden, conv, recurrent):
        actual = adapters[mode](hidden, conv, recurrent)
        reference = native_call(hidden, mode == "decode", conv, recurrent)
        compare(label, actual, reference)
        onnx_cases.append((label, mode, (hidden, conv, recurrent), reference))
        return actual, reference

    try:
        with torch.no_grad():
            for length in lane["cpuLengths"]:
                hidden = torch.randn(1, length, native.hidden_size)
                conv, recurrent = torch.randn(conv_shape), torch.randn(recurrent_shape)
                check_case(f"prefill-{length}", "prefill", hidden, conv, recurrent)
            hidden = torch.randn(1, 1, native.hidden_size)
            conv, recurrent = torch.randn(conv_shape), torch.randn(recurrent_shape)
            check_case("decode-random-state", "decode", hidden, conv, recurrent)
            for partition in lane["cpuPartitions"]:
                hidden = torch.randn(1, sum(partition), native.hidden_size)
                native_states = (torch.zeros(conv_shape), torch.zeros(recurrent_shape))
                export_states = tuple(x.clone() for x in native_states)
                offset = 0
                for step, length in enumerate(partition):
                    piece = hidden[:, offset:offset + length]
                    mode = "prefill" if step == 0 else "decode"
                    actual = adapters[mode](piece, *export_states)
                    reference = native_call(piece, step > 0, *native_states)
                    compare(f"partition-{partition}-step-{step}", actual, reference)
                    onnx_cases.append((f"partition-{partition}-step-{step}", mode,
                                       (piece, *native_states), reference))
                    export_states, native_states = actual[1:], reference[1:]
                    offset += length
            receipt["cpuParity"] = "passed"
            if args.export:
                import onnx
                for mode in ("prefill", "decode"):
                    length = 2 if mode == "prefill" else 1
                    dynamic = {"hidden": {1: torch.export.Dim("sequence_length", min=2, max=lane["prefillBound"])},
                               "past_conv": {}, "past_recurrent": {}} if mode == "prefill" else None
                    exported = torch.export.export(adapters[mode], (torch.zeros(1, length, native.hidden_size),
                                                   torch.zeros(conv_shape), torch.zeros(recurrent_shape)),
                                                   dynamic_shapes=dynamic, strict=True)
                    program = torch.onnx.export(exported, dynamo=True, optimize=True,
                        input_names=["hidden", "past_conv", "past_recurrent"],
                        output_names=["output", "present_conv", "present_recurrent"],
                        opset_version=policy["exportFrontend"]["opset"])
                    target = args.out / (mode + ".onnx")
                    program.save(target)
                    graph = onnx.load(target, load_external_data=False)
                    lowerings = lower_rms_normalization(graph)
                    onnx.save(graph, target)
                    receipt["exports"].append({"mode": mode, "sha256": digest(target),
                        "lowerings": lowerings,
                        "operators": dict(sorted(Counter(n.op_type for n in graph.graph.node).items())),
                        "hiddenShape": [str(d) for d in program.model.graph.inputs[0].shape]})
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = policy["threads"]
                options.inter_op_num_threads = policy["threads"]
                sessions = {mode: ort.InferenceSession(str(args.out / (mode + ".onnx")), options,
                                                       providers=["CPUExecutionProvider"])
                            for mode in ("prefill", "decode")}
                for label, mode, arguments, reference in onnx_cases:
                    feeds = {name: value.numpy() for name, value in zip(
                        ("hidden", "past_conv", "past_recurrent"), arguments)}
                    result = sessions[mode].run(None, feeds)
                    compare("onnx-cpu-" + label, tuple(torch.from_numpy(v) for v in result), reference)
                receipt["onnxCpuParity"] = "passed"
                fixtures = []
                for index, (label, mode, arguments, reference) in enumerate(onnx_cases):
                    payload, tensors = bytearray(), []
                    for role, names, values in (("input", ("hidden", "past_conv", "past_recurrent"), arguments),
                                                 ("expected", ("output", "present_conv", "present_recurrent"), reference)):
                        for name, value in zip(names, values):
                            raw = value.detach().numpy().astype("<f4", copy=False).tobytes()
                            tensors.append({"role": role, "name": name, "shape": list(value.shape),
                                            "offset": len(payload), "bytes": len(raw), "dtype": "float32"})
                            payload.extend(raw)
                    filename = f"case-{index}.bin"
                    (args.out / filename).write_bytes(payload)
                    fixtures.append({"label": label, "mode": mode, "file": filename,
                                     "sha256": hashlib.sha256(payload).hexdigest(), "tensors": tensors})
                (args.out / "fixtures.json").write_text(json.dumps({"schema": "doppler.qwen35-linear-fixtures.v1",
                    "sourceReceipt": "receipt.json", "absoluteTolerance": lane["absoluteTolerance"],
                    "relativeTolerance": lane["relativeTolerance"], "cases": fixtures}, indent=2) + "\n")
        receipt["completed"] = True
    except Exception as error:
        receipt["completed"] = False
        receipt["error"] = {"exception": type(error).__name__, "message": str(error)}
        (args.out / "failure.log").write_text(traceback.format_exc())
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cpuParity": receipt.get("cpuParity"), "completed": receipt["completed"],
                      "checks": len(receipt["checks"]), "exports": len(receipt["exports"]),
                      "error": receipt.get("error")}, indent=2), flush=True)
    if not receipt["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
