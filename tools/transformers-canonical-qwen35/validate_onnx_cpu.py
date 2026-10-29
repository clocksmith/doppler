"""Execute full ONNX sessions on CPU and compare every logit/state to native HF."""
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
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lane-policy", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    lane = json.loads(args.lane_policy.read_text()) if args.lane_policy else policy["textExport"]
    inputs = read_inputs(args.canonical, policy)
    verify_bytes(inputs, policy["hashChunkBytes"])
    exported = json.loads((args.model / "receipt.json").read_text())
    if exported.get("completed") is not True or exported["inputs"] != inputs["inputs"]:
        raise ValueError("ONNX artifact construction receipt incomplete or different canonical source")
    if exported["contract"] != lane:
        raise ValueError("explicit CPU lane policy differs from frozen ONNX artifact")
    for artifact in exported["artifacts"]:
        path = args.model / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]:
            raise ValueError("ONNX graph or external initializer artifact changed: " + str(path))
    initializer = exported["initializerBinding"]
    if not initializer["pass"] or digest(Path(initializer["path"])) != initializer["sha256"]:
        raise ValueError("ONNX initializer binding proof changed or incomplete")
    expected = policy["frameworks"] | policy["exportFrontend"]["frameworks"]
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise ValueError("ONNX CPU numerical dependency mismatch")
    import torch
    import numpy as np
    import onnxruntime as ort
    from export_text import cache_names, cache_tensors, empty_cache

    torch.set_num_threads(policy["threads"])
    torch.set_num_interop_threads(policy["threads"])
    native, config, bindings = load_native_model(inputs, policy)
    teacher = json.loads(args.teacher.read_text())
    if (teacher.get("completed") is not True or teacher["inputs"] != inputs["inputs"] or
            teacher["contract"]["kvPhysicalDtype"] != lane["kvPhysicalDtype"]):
        raise ValueError("native teacher physical-cache lane or canonical identity differs")
    code = next(p for p in teacher["prompts"] if p["id"] == "code")
    options = ort.SessionOptions()
    options.intra_op_num_threads = policy["threads"]
    options.inter_op_num_threads = policy["threads"]
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_mem_pattern = False
    session_files = exported.get("sessionFiles")
    if session_files is None:
        if "sourceF16Artifact" in exported:
            raise ValueError("derived graph variant is missing explicit session paths")
        session_files = {name: str(args.model / "onnx" / (name + ".onnx"))
                         for name in ("embed_tokens", "decoder_prefill", "decoder_decode")}
    sessions = {name: ort.InferenceSession(session_files[name], options,
                                          providers=["CPUExecutionProvider"])
                for name in ("embed_tokens", "decoder_prefill", "decoder_decode")}
    state_names = cache_names(config)
    names = ["logits"] + [n for p, n in state_names]
    receipt = {"schema": "doppler.canonical-text-onnx-cpu.v1", "completed": False,
               "scope": "ONNX CPU numerical execution only; no TJS, GPU or speed proof", "inputs": inputs["inputs"],
               "frameworks": versions, "parameterCount": len(bindings), "policySha256": digest(policy_path), "contract": lane,
               "onnxArtifact": {"path": str(args.model / "receipt.json"), "sha256": digest(args.model / "receipt.json")},
               "teacher": {"path": str(args.teacher), "sha256": digest(args.teacher)},
               "provider": {"executionProviders": ["CPUExecutionProvider"], "graphOptimization": "disabled",
                            "threads": policy["threads"], "interOpThreads": policy["threads"]},
               "sourceSha256": {name: digest(Path(__file__).with_name(name)) for name in
                                  ("validate_onnx_cpu.py", "native_model.py", "export_text.py")},
               "checks": [], "cases": []}

    def compare(label, actual, reference):
        results = []
        for name, value, target in zip(names, actual, reference):
            expected_value = target.numpy()
            if value.shape != expected_value.shape or value.dtype != expected_value.dtype:
                raise ValueError("ONNX state shape/dtype differs from native: " + label + " " + name)
            difference = np.abs(value.astype(np.float32) - expected_value.astype(np.float32))
            maximum, mean = float(difference.max()), float(difference.mean())
            finite = bool(np.isfinite(value).all() and np.isfinite(expected_value).all())
            if name == "logits":
                passed = finite and maximum <= lane["logitAbsoluteTolerance"]
                extra = {"argmax": int(value.argmax()), "nativeArgmax": int(expected_value.argmax())}
            elif value.dtype == np.float16:
                def ordered(array):
                    bits = array.view(np.uint16).astype(np.int32)
                    return np.where(bits >= 32768, 32768 - (bits & 32767), 32768 + bits)
                steps = int(np.abs(ordered(value) - ordered(expected_value)).max())
                passed = finite and steps <= lane["kvStateMaxRepresentableSteps"]
                extra = {"maxRepresentableSteps": steps}
            else:
                limit = lane["stateAbsoluteTolerance"] + lane["stateRelativeTolerance"] * np.abs(expected_value)
                passed = finite and bool((difference <= limit).all())
                extra = {}
            check = {"case": label, "name": name, "dtype": str(value.dtype), "shape": list(value.shape),
                     "maxAbsoluteError": maximum, "meanAbsoluteError": mean, "pass": passed, **extra}
            receipt["checks"].append(check)
            results.append(passed)
        return all(results)

    try:
        with torch.inference_mode():
            for length in lane["cpuPrefillLengths"]:
                ids = torch.tensor([code["promptIds"][:length]], dtype=torch.int64)
                physical_dtype = {"float16": torch.float16, "float32": torch.float32}[lane["kvPhysicalDtype"]]
                native_cache = physical_cache(config, physical_dtype)
                onnx_cache = tuple((t.to(physical_dtype) if config.layer_types[index // 2] == "full_attention" else t).numpy()
                                   for index, t in enumerate(empty_cache(config)))
                for step in range(lane["cpuDecodeSteps"] + 1):
                    mode = "prefill" if step == 0 else "decode"
                    offset = 0 if step == 0 else length + step - 1
                    metadata = {value.name: value.type for value in sessions["decoder_" + mode].get_inputs()}
                    integer_types = {"tensor(int64)": np.int64, "tensor(int32)": np.int32}
                    positions = np.broadcast_to(np.arange(offset, offset + ids.shape[1], dtype=integer_types[metadata["position_ids"]]),
                                                (3, 1, ids.shape[1])).copy()
                    mask = np.ones((1, offset + ids.shape[1]), dtype=integer_types[metadata["attention_mask"]])
                    embedded = sessions["embed_tokens"].run(None, {"input_ids": ids.numpy()})[0]
                    if not np.array_equal(embedded, native.model.embed_tokens(ids).numpy()):
                        raise ValueError("ONNX embedding output is not byte-exact canonical lookup")
                    feeds = {"inputs_embeds": embedded, "position_ids": positions, "attention_mask": mask,
                             **{pair[0]: value for pair, value in zip(state_names, onnx_cache)}}
                    for control in lane.get("hostGeometryInputs", {}).get(mode, []):
                        recipes = control["values"] if "values" in control else [control]
                        values = [expression["constant"] if "constant" in expression else
                                  feeds[expression["sourceTensor"]].shape[expression["axis"]] * expression["multiplier"] + expression["addend"]
                                  for expression in recipes]
                        feeds[control["name"]] = np.array(values, dtype={"int32": np.int32, "int64": np.int64}[control["type"]]).reshape(control["shape"])
                    actual = sessions["decoder_" + mode].run(None, feeds)
                    reference_output = native(input_ids=ids, past_key_values=native_cache, use_cache=True, logits_to_keep=1)
                    inspect_cache(native_cache, config)
                    reference = (reference_output.logits, *cache_tensors(native_cache, config))
                    label = f"prefill-{length}-{mode}-{step}"
                    passed = compare(label, actual, reference)
                    token = int(reference_output.logits[0, -1].argmax())
                    artifacts = []
                    for engine, value in (("onnx", actual[0]), ("native", reference[0].numpy())):
                        path = args.out / (label + "-" + engine + ".f32")
                        path.write_bytes(value.astype("<f4", copy=False).tobytes())
                        artifacts.append({"engine": engine, "file": path.name, "sha256": digest(path)})
                    receipt["cases"].append({"name": label, "inputIds": ids.flatten().tolist(), "positionOffset": offset,
                        "nativeArgmax": token, "onnxArgmax": int(actual[0].argmax()), "pass": passed, "logits": artifacts})
                    onnx_cache = actual[1:]
                    ids = torch.tensor([[token]], dtype=torch.int64)
                    print(json.dumps({"case": label, "pass": passed}), flush=True)
        receipt["completed"] = True
        receipt["pass"] = all(c["pass"] for c in receipt["checks"])
    except Exception as error:
        receipt["error"] = {"exception": type(error).__name__, "message": str(error)}
        (args.out / "failure.log").write_text(traceback.format_exc())
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["completed"] or not receipt.get("pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
