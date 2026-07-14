#!/usr/bin/env python3
"""Capture Qwen 3.5 Transformers layer boundaries for Doppler diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl_row(path: Path, row_index: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if row_index < 0 or row_index >= len(rows):
        raise ValueError(f"row index {row_index} is outside [0, {len(rows)})")
    row = rows[row_index]
    if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
        raise ValueError("selected JSONL row must contain a string prompt")
    return row


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    flat = value.reshape(-1)
    return {
        "shape": list(value.shape),
        "sourceDtype": str(tensor.dtype).removeprefix("torch."),
        "dtype": "float32",
        "elementCount": int(flat.numel()),
        "sample": [float(item) for item in flat[:8].tolist()],
        "stats": {
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "maxAbs": float(flat.abs().max().item()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
        },
        "finite": bool(torch.isfinite(flat).all().item()),
    }


def output_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)) and value and hasattr(value[0], "detach"):
        return value[0]
    raise TypeError(f"hook output does not contain a tensor: {type(value).__name__}")


def resolve_text_model(model: Any) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
            return candidate
    raise RuntimeError("unable to resolve Qwen 3.5 text model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    model_path = args.model.resolve()
    dataset_path = args.dataset.resolve()
    output_path = args.out.resolve()
    row = read_jsonl_row(dataset_path, args.row_index)
    prompt = row["prompt"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    text_model = resolve_text_model(model)
    if args.layer < 0 or args.layer >= len(text_model.layers):
        raise ValueError(f"layer {args.layer} is outside [0, {len(text_model.layers)})")
    layer = text_model.layers[args.layer]
    if getattr(layer, "block_type", None) != "linear_attention":
        raise ValueError(f"layer {args.layer} is not a linear_attention layer")

    captures: dict[str, dict[str, Any]] = {}
    handles = []

    def capture_output(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captures[name] = tensor_summary(output_tensor(output))

        return hook

    def capture_input(name: str):
        def hook(_module: Any, inputs: Any) -> None:
            captures[name] = tensor_summary(output_tensor(inputs))

        return hook

    module_outputs = {
        "embed.out": text_model.embed_tokens,
        f"layer.{args.layer}.attn.qkv_proj": layer.linear_attn.in_proj_qkv,
        f"layer.{args.layer}.attn.linear_z_proj": layer.linear_attn.in_proj_z,
        f"layer.{args.layer}.attn.linear_a_proj": layer.linear_attn.in_proj_a,
        f"layer.{args.layer}.attn.linear_b_proj": layer.linear_attn.in_proj_b,
        f"layer.{args.layer}.attn.linear_core_out": layer.linear_attn.norm,
        f"layer.{args.layer}.attn.out": layer.linear_attn.out_proj,
        f"layer.{args.layer}.ffn.in": layer.post_attention_layernorm,
        f"layer.{args.layer}.ffn.gate": layer.mlp.gate_proj,
        f"layer.{args.layer}.ffn.up": layer.mlp.up_proj,
        f"layer.{args.layer}.ffn.out": layer.mlp.down_proj,
        f"layer.{args.layer}.layer.out": layer,
    }
    for name, module in module_outputs.items():
        handles.append(module.register_forward_hook(capture_output(name)))
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
        capture_input(f"layer.{args.layer}.attn.residual_out")
    ))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(
        capture_input(f"layer.{args.layer}.ffn.act")
    ))

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        selected_token_id = int(outputs.logits[0, -1].float().argmax().item())
    finally:
        for handle in handles:
            handle.remove()

    ordered_names = [name for name in module_outputs if name in captures]
    residual_name = f"layer.{args.layer}.attn.residual_out"
    if residual_name in captures:
        ordered_names.insert(7, residual_name)
    activation_name = f"layer.{args.layer}.ffn.act"
    if activation_name in captures:
        ordered_names.insert(11, activation_name)
    receipt = {
        "schema": "doppler.qwen35-hf-boundary-reference/v1",
        "ok": True,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": str(torch.version.hip),
            "transformers": transformers.__version__,
            "transformersModulePath": transformers.__file__,
            "device": torch.cuda.get_device_name(0),
            "dtype": args.dtype,
        },
        "model": {
            "path": str(model_path),
            "revision": args.model_revision,
            "configSha256": sha256_file(model_path / "config.json"),
            "tokenizerSha256": sha256_file(model_path / "tokenizer.json"),
        },
        "probe": {
            "datasetPath": str(dataset_path),
            "datasetSha256": sha256_file(dataset_path),
            "rowIndex": args.row_index,
            "rowId": row.get("rowId") or row.get("id"),
            "promptSha256": sha256_text(prompt),
            "promptTokenIds": [int(item) for item in input_ids[0].tolist()],
            "useChatTemplate": False,
        },
        "layer": args.layer,
        "selectedTokenId": selected_token_id,
        "captures": [{"opId": name, **captures[name]} for name in ordered_names],
        "claimBoundary": (
            "Transformers boundary diagnostics only; this does not establish Doppler parity, "
            "adapter portability, checkpoint selection, semantic correctness, or promotion."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
