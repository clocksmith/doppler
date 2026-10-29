"""Independent native HF CPU reference, with explicit physical KV cache dtype."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from binding import bind_names, digest, read_inputs, verify_bytes


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
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    import transformers.cache_utils as cache_module

    if hf.FusedRMSNormGated is not None or hf.is_fast_path_available:
        raise ValueError("native CPU reference refuses optional CUDA kernels")
    torch.set_num_threads(policy["threads"])
    torch.set_num_interop_threads(policy["threads"])
    config = Qwen3_5TextConfig(**inputs["config"]["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    mappings = bind_names(inputs["manifest"]["tensors"],
                          {k: list(t.shape) for k, t in model.state_dict().items()}, policy)
    with safe_open(str(inputs["weights"]), framework="pt", device="cpu") as source:
        state = {m["target"]: source.get_tensor(m["source"]) for m in mappings}
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # These deterministic native nonpersistent RoPE buffers are not checkpoint parameters.
    model.model.rotary_emb = hf.Qwen3_5TextRotaryEmbedding(config, device="cpu")
    model.eval()
    if model.lm_head.weight is model.model.embed_tokens.weight or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        raise ValueError("canonical LM head was aliased to input embeddings")
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in model.parameters()):
        raise ValueError("native reference parameters must all be CPU F32")
    if any(t.device.type != "cpu" for t in model.buffers()):
        raise ValueError("native reference retained a non-CPU buffer")
    kv_dtype = torch.float16 if args.kv_dtype == "f16" else torch.float32

    class PhysicalKVCache(DynamicCache):
        def __init__(self):
            super().__init__(config=config)
            self.events = []

        def update(self, key_states, value_states, layer_idx, *extra, **kwargs):
            if key_states.dtype != torch.float32 or value_states.dtype != torch.float32:
                raise ValueError("eager attention K/V producer must be F32 before declared cache rounding")
            keys, values = super().update(key_states.to(kv_dtype), value_states.to(kv_dtype),
                                          layer_idx, *extra, **kwargs)
            layer = self.layers[layer_idx]
            if (keys.dtype != kv_dtype or values.dtype != kv_dtype or
                    layer.keys.dtype != kv_dtype or layer.values.dtype != kv_dtype):
                raise ValueError("actual cache storage differs from declared physical dtype")
            widened = keys.float(), values.float()
            if not (torch.equal(widened[0].to(kv_dtype), keys) and torch.equal(widened[1].to(kv_dtype), values)):
                raise ValueError("widened attention inputs are not exact cache values")
            self.events.append({"layer": layer_idx, "newLength": key_states.shape[2],
                                "storedShape": list(keys.shape), "producerDtype": str(key_states.dtype),
                                "physicalKeyDtype": str(layer.keys.dtype), "physicalValueDtype": str(layer.values.dtype),
                                "attentionKeyDtype": str(widened[0].dtype), "attentionValueDtype": str(widened[1].dtype),
                                "newKeyRoundingMax": float((key_states - key_states.to(kv_dtype).float()).abs().max()),
                                "newValueRoundingMax": float((value_states - value_states.to(kv_dtype).float()).abs().max())})
            return widened

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
                                 "reference": digest(Path(__file__))},
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
            cache = PhysicalKVCache()
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
            full_layers = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
            if [e["layer"] for e in cache.events] != full_layers:
                raise ValueError("native reference did not observe all full-attention cache updates")
            states = []
            for index, kind in enumerate(config.layer_types):
                layer = cache.layers[index]
                if kind == "linear_attention":
                    if layer.conv_states.dtype != torch.float32 or layer.recurrent_states.dtype != torch.float32:
                        raise ValueError("native linear state is not F32")
                    states.append({"layer": index, "convDtype": str(layer.conv_states.dtype),
                                   "recurrentDtype": str(layer.recurrent_states.dtype)})
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
