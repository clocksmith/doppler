"""Inspect canonical F32 bindings against the pinned HF meta-device text model."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

from binding import bind_names, digest, read_inputs, verify_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify-bytes", action="store_true")
    args = parser.parse_args()
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    inputs = read_inputs(args.canonical, policy)
    versions = {name: importlib.metadata.version(name) for name in policy["frameworks"]}
    if versions != policy["frameworks"]:
        raise ValueError("export framework versions differ: " + repr(versions))
    if args.verify_bytes:
        verify_bytes(inputs, policy["hashChunkBytes"])

    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5, configuration_qwen3_5

    if modeling_qwen3_5.FusedRMSNormGated is not None:
        raise ValueError("meta-only audit refuses a constructor with a CUDA fused-normalization hook")
    torch.set_num_threads(policy["threads"])
    config = Qwen3_5TextConfig(**inputs["config"]["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    state = model.state_dict()
    if any(t.device.type != "meta" for t in state.values()):
        raise ValueError("metadata audit allocated a non-meta model parameter")
    if model.lm_head.weight is model.model.embed_tokens.weight:
        raise ValueError("HF constructor tied the canonical output head")
    mappings = bind_names(inputs["manifest"]["tensors"],
                          {name: list(t.shape) for name, t in state.items()}, policy)
    caches = []
    text = inputs["config"]["text_config"]
    for layer, kind in enumerate(text["layer_types"]):
        if kind == "linear_attention":
            conv_width = (2 * text["linear_num_key_heads"] * text["linear_key_head_dim"] +
                          text["linear_num_value_heads"] * text["linear_value_head_dim"])
            pairs = [("conv", ["batch_size", conv_width, text["linear_conv_kernel_dim"]]),
                     ("recurrent", ["batch_size", text["linear_num_value_heads"],
                                    text["linear_key_head_dim"], text["linear_value_head_dim"]])]
            for role, shape in pairs:
                caches.append({"input": f"past_{role}.{layer}", "output": f"present_{role}.{layer}",
                               "shape": shape, "dtype": policy["dtype"]})
        elif kind == "full_attention":
            for role in ("key", "value"):
                caches.append({"input": f"past_key_values.{layer}.{role}",
                               "output": f"present.{layer}.{role}",
                               "shape": ["batch_size", text["num_key_value_heads"],
                                         "past_sequence_length", text["head_dim"]],
                               "dtype": policy["dtype"]})
        else:
            raise ValueError("unsupported layer type: " + kind)
    modules = [torch, modeling_qwen3_5, configuration_qwen3_5]
    receipt = {"schema": "doppler.transformers-canonical-binding.v1",
               "scope": "CPU/meta structural compatibility only; no ONNX artifact or inference proof",
               "inputs": inputs["inputs"], "policySha256": digest(policy_path),
               "frameworks": versions, "byteVerification": "passed" if args.verify_bytes else "not-run",
               "frameworkSources": {str(Path(inspect.getfile(m))): digest(Path(inspect.getfile(m)))
                                    for m in modules},
               "parameters": mappings, "stateCache": caches,
               "semantics": {"head": "separate untied canonical matrix",
                             "A_log": "raw stored parameter; HF computes -exp(A_log) in forward",
                             "rmsNorm": "HF Qwen3_5RMSNorm multiplies by 1 + stored weight",
                             "gatedNorm": "HF gated linear-attention norm uses its stored weight directly"},
               "onnxExport": "not-attempted", "runnable": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("refusing to overwrite an existing binding receipt")
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"parameters": len(mappings), "cacheTensors": len(caches),
                      "byteVerification": receipt["byteVerification"], "runnable": False,
                      "receipt": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
