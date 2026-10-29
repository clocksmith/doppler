"""Full24-layer exporter CPU parity against unchanged native HF and F16 KV."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

from binding import digest, read_inputs, verify_bytes
from native_model import load_native_model, physical_cache, inspect_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--diagnose-softplus", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    lane = policy["textExport"]
    inputs = read_inputs(args.canonical, policy)
    verify_bytes(inputs, policy["hashChunkBytes"])
    expected = policy["frameworks"]
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise ValueError("native comparison framework mismatch")
    import torch
    from export_text import TextDecoderExport, empty_cache, cache_tensors, cache_names
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    torch.set_num_threads(policy["threads"])
    torch.set_num_interop_threads(policy["threads"])
    native, config, mappings = load_native_model(inputs, policy)
    teacher = json.loads(args.teacher.read_text())
    if teacher.get("schema") != "doppler.native-hf-canonical-teacher.v1" or teacher.get("completed") is not True:
        raise ValueError("full-model CPU validation requires completed independent teacher")
    if teacher["inputs"] != inputs["inputs"]:
        raise ValueError("teacher canonical identity differs")
    code = next(p for p in teacher["prompts"] if p["id"] == "code")
    adapters = {mode: TextDecoderExport(native, lane, mode).eval() for mode in ("prefill", "decode")}
    softplus_probe = []
    if args.diagnose_softplus:
        from export_layer import LinearAttentionExport
        gate = native.model.layers[5].linear_attn

        def capture_gate(module, arguments, result):
            candidate = result.detach() + gate.dt_bias
            native_value = torch.nn.functional.softplus(candidate)
            export_value = LinearAttentionExport.softplus(candidate)
            error = (native_value - export_value).abs()
            softplus_probe.append({"call": len(softplus_probe), "input": candidate.flatten().tolist(),
                                   "native": native_value.flatten().tolist(), "adapter": export_value.flatten().tolist(),
                                   "maxAbsoluteError": float(error.max()), "differentElements": int((error != 0).sum())})
        gate.in_proj_a.register_forward_hook(capture_gate)
    names = ["logits"] + [present for past, present in cache_names(config)]
    receipt = {"schema": "doppler.canonical-text-export-cpu.v1", "completed": False,
               "scope": "Full24-layer CPU adapter parity; not ONNX, TJS, GPU or performance proof",
               "inputs": inputs["inputs"], "parameterCount": len(mappings), "frameworks": versions,
               "contract": lane, "policySha256": digest(policy_path),
               "teacher": {"path": str(args.teacher), "sha256": digest(args.teacher)},
               "sourceSha256": {name: digest(Path(__file__).with_name(name)) for name in
                    ("validate_text.py", "export_text.py", "export_layer.py", "native_model.py")},
               "nativeSourceSha256": digest(Path(hf.__file__)), "checks": [], "cases": []}

    def compare(label, actual, reference):
        if len(actual) != 49 or len(reference) != 49:
            raise ValueError("full decoder must expose logits and48state tensors")
        for name, value, expected_value in zip(names, actual, reference):
            if value.shape != expected_value.shape or value.dtype != expected_value.dtype:
                raise ValueError(f"full decoder state shape/dtype mismatch: {label} {name}")
            error = (value.float() - expected_value.float()).abs()
            maximum = float(error.max()) if error.numel() else 0.0
            mean = float(error.mean()) if error.numel() else 0.0
            finite = bool(torch.isfinite(value).all() and torch.isfinite(expected_value).all())
            if name == "logits":
                passed = finite and maximum <= lane["logitAbsoluteTolerance"]
                extra = {"argmax": int(value.flatten().argmax()), "nativeArgmax": int(expected_value.flatten().argmax())}
            elif value.dtype == torch.float16:
                def ordered(tensor):
                    bits = tensor.view(torch.int16).to(torch.int32) & 65535
                    return torch.where(bits >= 32768, 32768 - (bits & 32767), 32768 + bits)
                distance = (ordered(value) - ordered(expected_value)).abs()
                steps = int(distance.max()) if distance.numel() else 0
                passed = finite and steps <= lane["kvStateMaxRepresentableSteps"]
                extra = {"maxRepresentableSteps": steps}
            else:
                limit = lane["stateAbsoluteTolerance"] + lane["stateRelativeTolerance"] * expected_value.abs()
                passed = finite and bool((error <= limit).all())
                extra = {}
            check = {"case": label, "name": name, "dtype": str(value.dtype), "shape": list(value.shape),
                     "maxAbsoluteError": maximum, "meanAbsoluteError": mean, "pass": passed, **extra}
            receipt["checks"].append(check)
            if not passed:
                raise ValueError("full decoder native CPU parity failed: " + json.dumps(check))

    try:
        with torch.inference_mode():
            for length in lane["cpuPrefillLengths"]:
                ids = torch.tensor([code["promptIds"][:length]], dtype=torch.int64)
                native_cache = physical_cache(config, torch.float16)
                exported_cache = empty_cache(config)
                for step in range(lane["cpuDecodeSteps"] + 1):
                    mode = "prefill" if step == 0 else "decode"
                    offset = 0 if step == 0 else length + step - 1
                    positions = torch.arange(offset, offset + ids.shape[1]).reshape(1, 1, -1).expand(3, 1, -1)
                    attention_mask = torch.ones(1, offset + ids.shape[1], dtype=torch.int64)
                    actual = adapters[mode](native.model.embed_tokens(ids), positions, attention_mask, exported_cache)
                    expected_result = native(input_ids=ids, past_key_values=native_cache, use_cache=True, logits_to_keep=1)
                    inspect_cache(native_cache, config)
                    reference = (expected_result.logits, *cache_tensors(native_cache, config))
                    label = f"prefill-{length}-{mode}-{step}"
                    compare(label, actual, reference)
                    token = int(expected_result.logits[0, -1].argmax())
                    receipt["cases"].append({"name": label, "inputIds": ids.flatten().tolist(), "positionOffset": offset,
                                             "nativeArgmax": token, "exportArgmax": int(actual[0][0, -1].argmax())})
                    exported_cache = actual[1:]
                    ids = torch.tensor([[token]], dtype=torch.int64)
                    print(json.dumps({"case": label, "pass": True}), flush=True)
        receipt["completed"] = True
    except Exception as error:
        receipt["error"] = {"exception": type(error).__name__, "message": str(error)}
        (args.out / "failure.log").write_text(traceback.format_exc())
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.diagnose_softplus:
        (args.out / "softplus-diagnostic.json").write_text(json.dumps(softplus_probe, indent=2) + "\n")
    if not receipt["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
