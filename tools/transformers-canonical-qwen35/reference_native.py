"""Independent native HF CPU reference, with explicit physical KV cache dtype."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from binding import digest, read_inputs, verify_bytes
from native_model import load_native_model, physical_cache, inspect_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--pooled", required=True, type=Path)
    parser.add_argument("--doppler", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--kv-dtype", required=True, choices=("f16", "f32"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    inputs = read_inputs(args.canonical, policy)
    verify_bytes(inputs, policy["hashChunkBytes"])
    versions = {name: importlib.metadata.version(name) for name in policy["frameworks"]}
    if versions != policy["frameworks"]:
        raise ValueError("native reference framework identity mismatch")

    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    import transformers.cache_utils as cache_module

    torch.set_num_threads(policy["threads"])
    torch.set_num_interop_threads(policy["threads"])
    model, config, mappings = load_native_model(inputs, policy)
    kv_dtype = torch.float16 if args.kv_dtype == "f16" else torch.float32

    pooled = json.loads(args.pooled.read_text())
    doppler = json.loads(args.doppler.read_text())["result"]["metrics"]
    prompt = pooled["promptIds"]
    transcript_ids = doppler["referenceTranscript"]["prompt"]["ids"]
    if prompt != transcript_ids:
        raise ValueError("parent engine prompt token IDs differ")
    receipt = {"schema": "doppler.native-hf-canonical-reference.v1",
               "scope": "CPU native HF eager F32 math with explicit physical cache rounding; not a GPU or speed receipt",
               "inputs": inputs["inputs"], "parameters": mappings, "byteVerification": "passed",
               "frameworks": versions, "policySha256": digest(policy_path),
               "sourceSha256": {"native": digest(Path(hf.__file__)), "cache": digest(Path(cache_module.__file__)),
                                 "reference": digest(Path(__file__)),
                                 "modelBoundary": digest(Path(__file__).with_name("native_model.py"))},
               "engineReceipts": {"pooled": {"path": str(args.pooled), "sha256": digest(args.pooled)},
                                  "doppler": {"path": str(args.doppler), "sha256": digest(args.doppler)}},
               "contract": {"weights": "F32", "kvPhysical": args.kv_dtype,
                            "attentionMath": "F32 on exactly widened physical cache values",
                            "recurrentState": "F32", "convolutionState": "F32", "threads": policy["threads"],
                            "head": "separate untied canonical matrix"},
               "promptIds": prompt, "runs": [], "comparisons": []}

    def compare(label, observed, engine):
        reference = torch.tensor(engine, dtype=torch.float32)
        value = observed.flatten()
        if value.shape != reference.shape or not torch.isfinite(value).all():
            raise ValueError("invalid comparison shape/finiteness: " + label)
        error = (value - reference).abs()
        return {"name": label, "maxAbsoluteError": float(error.max()), "meanAbsoluteError": float(error.mean()),
                "p99AbsoluteError": float(torch.quantile(error, 0.99)),
                "nativeArgmax": int(value.argmax()), "engineArgmax": int(reference.argmax())}

    with torch.inference_mode():
        for name, ids in (("prompt", prompt), ("single", prompt[:1])):
            cache = physical_cache(config, kv_dtype)
            boundaries = []
            hooks = []
            if name == "single":
                for layer_index, layer in enumerate(model.model.layers):
                    def capture(module, arguments, result, index=layer_index):
                        boundaries.append({"layer": index, "values": result.flatten().tolist()})
                    hooks.append(layer.register_forward_hook(capture))
            try:
                result = model(input_ids=torch.tensor([ids], dtype=torch.int64), past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
            finally:
                for hook in hooks:
                    hook.remove()
            logits = result.logits[0, -1].float()
            states = inspect_cache(cache, config)
            receipt["runs"].append({"name": name, "cacheEvents": cache.events, "linearStates": states,
                                    "logits": logits.tolist(), "boundaries": boundaries})
            receipt["comparisons"].append(compare(name + "-pooled", logits,
                                                   pooled["firstLogits" if name == "prompt" else "singleTokenLogits"]))
            receipt["comparisons"].append(compare(name + "-doppler", logits,
                                                   doppler["canonicalFirstLogits" if name == "prompt" else "canonicalSingleTokenLogits"]))
            if name == "single":
                for boundary in boundaries:
                    index, values = boundary["layer"], torch.tensor(boundary["values"])
                    matches = [b for b in pooled["singleTokenLayers"] if b["layer"] == index]
                    for ordinal, other in enumerate(matches):
                        receipt["comparisons"].append(compare(f"layer-{index}-pooled-{ordinal}", values, other["values"]))
                    matches = [b for b in doppler["canonicalSingleTokenBoundaries"]
                               if b["layer"] == index and b["stage"] == "layer_out"]
                    for ordinal, other in enumerate(matches):
                        receipt["comparisons"].append(compare(f"layer-{index}-doppler-{ordinal}", values, other["values"]))
            print(json.dumps({"run": name, "argmax": int(logits.argmax()), "cacheUpdates": len(cache.events)}), flush=True)
    receipt["completed"] = True
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["comparisons"][:4], indent=2), flush=True)


if __name__ == "__main__":
    main()
