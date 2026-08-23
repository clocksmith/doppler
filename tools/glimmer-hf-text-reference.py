#!/usr/bin/env python3
"""Run a pinned, text-only Muse Glimmer Transformers oracle from raw checkpoint truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "doppler.pinned-transformers-text-reference/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def read_policy(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require_equal(value.get("schema"), SCHEMA, "reference policy schema")
    require_equal(value.get("execution", {}).get("device"), "cpu", "reference device")
    if value.get("execution", {}).get("dtype") not in {"bfloat16", "float32"}:
        raise ValueError("reference dtype must be bfloat16 or float32")
    require_equal(value.get("execution", {}).get("attentionImplementation"), "eager", "attention implementation")
    if value.get("generation", {}).get("sampling") != "greedy-argmax-f32-logits":
        raise ValueError("reference sampling must be deterministic greedy argmax over f32 logits")
    if not isinstance(value.get("generation", {}).get("maxNewTokens"), int):
        raise ValueError("reference maxNewTokens must be an integer")
    boundary_steps = value.get("boundaryGenerationSteps")
    if (
        not isinstance(boundary_steps, list)
        or not boundary_steps
        or any(not isinstance(step, int) or step < 0 for step in boundary_steps)
        or boundary_steps != sorted(set(boundary_steps))
    ):
        raise ValueError("boundaryGenerationSteps must be a non-empty sorted set of non-negative integers")
    if boundary_steps[-1] >= value["generation"]["maxNewTokens"]:
        raise ValueError("boundaryGenerationSteps must remain within maxNewTokens")
    boundary_layers = value.get("boundaryLayers")
    if (
        not isinstance(boundary_layers, list)
        or not boundary_layers
        or any(not isinstance(layer, int) or layer < 0 for layer in boundary_layers)
        or boundary_layers != sorted(set(boundary_layers))
    ):
        raise ValueError("boundaryLayers must be a non-empty sorted set of non-negative integers")
    return value


def tensor_summary(
    tensor: Any,
    artifact_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if value.ndim >= 2 and value.shape[0] == 1:
        value = value.squeeze(0).contiguous()
    if value.ndim >= 2:
        value = value[-1].contiguous()
    flat = value.reshape(-1)
    array = value.numpy()
    summary = {
        "shape": list(value.shape),
        "dtype": "float32",
        "elementCount": int(flat.numel()),
        "samples": [float(item) for item in flat[:8].tolist()],
        "fullTensorDigest": "sha256:" + hashlib.sha256(array.tobytes()).hexdigest(),
        "statistics": {
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "maxAbs": float(flat.abs().max().item()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()),
        },
        "finite": bool(torch.isfinite(flat).all().item()),
    }
    if artifact_path is not None:
        if repo_root is None:
            raise ValueError("boundary artifact requires repo_root")
        import numpy

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.save(artifact_path, array, allow_pickle=False)
        summary["artifact"] = {
            "path": artifact_path.relative_to(repo_root).as_posix(),
            "fileSha256": "sha256:" + sha256_file(artifact_path),
            "payloadSha256": summary["fullTensorDigest"],
        }
    return summary


def output_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)) and value and hasattr(value[0], "detach"):
        return value[0]
    raise TypeError(f"hook output does not contain a tensor: {type(value).__name__}")


def assign_parameter(root: Any, name: str, tensor: Any) -> None:
    import torch

    parent_name, _, parameter_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, parameter_name, torch.nn.Parameter(tensor, requires_grad=False))


def load_text_only_model(model_dir: Path, config: Any, execution_dtype: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from safetensors import safe_open
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel

    class TextOnlyMuseGlimmer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = MuseGlimmerTextModel(config.text_config)
            self.lm_head = torch.nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )

    with torch.device("meta"):
        model = TextOnlyMuseGlimmer()
    expected = set(dict(model.named_parameters()).keys())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    absent = sorted(expected.difference(weight_map))
    if absent:
        raise ValueError(f"checkpoint is missing {len(absent)} text parameters: {absent[:3]}")

    loaded: set[str] = set()
    shard_names = sorted({weight_map[name] for name in expected})
    for shard_name in shard_names:
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            for name in sorted(expected):
                if weight_map[name] != shard_name:
                    continue
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"text parameter {name} must be BF16, got {tensor.dtype}")
                if execution_dtype == "float32":
                    tensor = tensor.to(dtype=torch.float32)
                assign_parameter(model, name, tensor)
                loaded.add(name)
    require_equal(loaded, expected, "loaded text parameter closure")

    rotary = model.model.language_model.rotary_emb
    inv_freq, _ = rotary.compute_default_rope_parameters(config.text_config, device="cpu")
    rotary.inv_freq = torch.nn.Buffer(inv_freq, persistent=False)
    rotary.original_inv_freq = torch.nn.Buffer(inv_freq.clone(), persistent=False)
    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
    if meta_parameters or meta_buffers:
        raise ValueError(f"reference model retained meta tensors: {meta_parameters[:3]} {meta_buffers[:3]}")
    model.eval()
    return model, {
        "loadedTextParameters": len(loaded),
        "preservedAuxiliaryParameters": len(weight_map) - len(loaded),
        "weightShardCount": len(shard_names),
        "sourceParameterDtype": "bfloat16",
        "executionParameterDtype": execution_dtype,
    }


def boundary_artifact_path(boundary_dir: Path, name: str) -> Path:
    return boundary_dir / (name.replace(".", "__") + ".npy")


def capture_boundaries(
    text_model: Any,
    layer_index: int,
    boundary_dir: Path,
    repo_root: Path,
    generation_step: int,
    include_embedding: bool,
) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    captures: dict[str, dict[str, Any]] = {}
    handles = []
    layer = text_model.layers[layer_index]
    phase = "prefill" if generation_step == 0 else "decode"
    prefix = "" if generation_step == 0 else f"generation.{generation_step}."
    modules = {
        f"layer.{layer_index}.input_norm": layer.input_layernorm,
        f"layer.{layer_index}.attention.output": layer.self_attn,
        f"layer.{layer_index}.post_attention_norm": layer.post_attention_layernorm,
        f"layer.{layer_index}.pre_ffn_norm": layer.pre_feedforward_layernorm,
        f"layer.{layer_index}.ffn.output": layer.mlp,
        f"layer.{layer_index}.post_ffn_norm": layer.post_feedforward_layernorm,
        f"layer.{layer_index}.output": layer,
        f"layer.{layer_index}.attention.q_proj": layer.self_attn.q_proj,
        f"layer.{layer_index}.attention.k_proj": layer.self_attn.k_proj,
        f"layer.{layer_index}.attention.v_proj": layer.self_attn.v_proj,
        f"layer.{layer_index}.attention.gate_proj": layer.self_attn.gate_proj,
        f"layer.{layer_index}.ffn.gate_proj": layer.mlp.gate_proj,
        f"layer.{layer_index}.ffn.up_proj": layer.mlp.up_proj,
        f"layer.{layer_index}.ffn.down_proj": layer.mlp.down_proj,
    }
    if include_embedding:
        modules["model.embedding.output"] = text_model.embed_tokens

    def hook(name: str):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            boundary_name = prefix + name
            captures[boundary_name] = {
                "phase": phase,
                "generationStep": generation_step,
                **tensor_summary(
                    output_tensor(output),
                    boundary_artifact_path(boundary_dir, boundary_name),
                    repo_root,
                ),
            }

        return capture

    for name, module in modules.items():
        handles.append(module.register_forward_hook(hook(name)))

    qk_norm_calls = 0

    def capture_qk_norm(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal qk_norm_calls
        base_name = (
            f"layer.{layer_index}.attention.q_norm"
            if qk_norm_calls == 0
            else f"layer.{layer_index}.attention.k_norm"
        )
        name = prefix + base_name
        value = output_tensor(output)
        value = value.transpose(1, 2).contiguous().reshape(value.shape[0], value.shape[2], -1)
        captures[name] = {
            "phase": phase,
            "generationStep": generation_step,
            **tensor_summary(
                value,
                boundary_artifact_path(boundary_dir, name),
                repo_root,
            ),
        }
        qk_norm_calls += 1

    def capture_o_proj_input(_module: Any, inputs: Any) -> None:
        name = prefix + f"layer.{layer_index}.attention.gated_output"
        captures[name] = {
            "phase": phase,
            "generationStep": generation_step,
            **tensor_summary(
                output_tensor(inputs),
                boundary_artifact_path(boundary_dir, name),
                repo_root,
            ),
        }

    handles.append(layer.self_attn.qk_norm.register_forward_hook(capture_qk_norm))
    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(capture_o_proj_input))
    return captures, handles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    policy_path = args.policy.resolve()
    policy = read_policy(policy_path)
    boundary_dir = (repo_root / policy["boundaryOutputDir"]).resolve()
    boundary_dir.relative_to(repo_root)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    for stale_boundary in boundary_dir.glob("*.npy"):
        stale_boundary.unlink()
    model_dir = (repo_root / policy["localModelDir"]).resolve()
    transformers_root = (repo_root / policy["transformersRoot"]).resolve()
    dependency_path = (repo_root / policy["dependencyPath"]).resolve()
    sys.path[:0] = [str(dependency_path), str(transformers_root / "src")]

    acquisition = json.loads((repo_root / policy["sourceAcquisitionReceipt"]).read_text(encoding="utf-8"))
    require_equal(acquisition.get("receiptDigest"), policy["sourceAcquisitionReceiptDigest"], "source acquisition")
    observed_commit = subprocess.check_output(
        ["git", "-C", str(transformers_root), "rev-parse", "HEAD"], text=True
    ).strip()
    require_equal(observed_commit, policy["transformersCommit"], "Transformers commit")
    for relative, expected_hash in policy["transformersFiles"].items():
        require_equal(sha256_file(transformers_root / relative), expected_hash, f"Transformers file {relative}")

    import torch
    import transformers
    from transformers import AutoTokenizer, MuseGlimmerConfig

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    config = MuseGlimmerConfig.from_pretrained(model_dir, local_files_only=True)
    generation_config = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    config.text_config._attn_implementation = policy["execution"]["attentionImplementation"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    encoded = tokenizer(
        policy["prompt"],
        return_tensors="pt",
        add_special_tokens=policy["generation"]["addSpecialTokens"],
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    model, load_evidence = load_text_only_model(model_dir, config, policy["execution"]["dtype"])
    text_model = model.model.language_model
    layer_indices = policy["boundaryLayers"]
    captures: dict[str, dict[str, Any]] = {}
    boundary_generation_steps = set(policy["boundaryGenerationSteps"])

    generated_ids: list[int] = []
    generation_steps: list[dict[str, Any]] = []
    past_key_values = None
    next_input = input_ids
    eos_ids = generation_config["eos_token_id"]
    eos_set = {int(item) for item in (eos_ids if isinstance(eos_ids, list) else [eos_ids])}
    for step in range(policy["generation"]["maxNewTokens"]):
        step_captures: dict[str, dict[str, Any]] = {}
        layer_capture_sets: list[dict[str, dict[str, Any]]] = []
        handles: list[Any] = []
        if step in boundary_generation_steps:
            for layer_offset, layer_index in enumerate(layer_indices):
                layer_captures, layer_handles = capture_boundaries(
                    text_model,
                    layer_index,
                    boundary_dir,
                    repo_root,
                    step,
                    include_embedding=layer_offset == 0,
                )
                layer_capture_sets.append(layer_captures)
                handles.extend(layer_handles)
        try:
            outputs = text_model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = model.lm_head(outputs.last_hidden_state[:, -1:, :]).float()
            logits = logits * config.text_config.output_multiplier
            logits = torch.tanh(logits / config.text_config.final_logit_softcapping)
            logits = logits * config.text_config.final_logit_softcapping
            if step in boundary_generation_steps:
                for layer_captures in layer_capture_sets:
                    step_captures.update(layer_captures)
                logits_name = "model.logits" if step == 0 else f"generation.{step}.model.logits"
                step_captures[logits_name] = {
                    "phase": "prefill" if step == 0 else "decode",
                    "generationStep": step,
                    **tensor_summary(
                        logits,
                        boundary_artifact_path(boundary_dir, logits_name),
                        repo_root,
                    ),
                }
                captures.update(step_captures)
        finally:
            for handle in handles:
                handle.remove()
        step_logits = logits[0, -1].detach().to(device="cpu", dtype=torch.float32).contiguous()
        top_values, top_ids = torch.topk(step_logits, k=8)
        next_token = int(top_ids[0].item())
        generated_ids.append(next_token)
        generation_steps.append({
            "index": step,
            "tokenId": next_token,
            "logitsDigest": "sha256:" + hashlib.sha256(step_logits.numpy().tobytes()).hexdigest(),
            "top": [
                {"tokenId": int(token_id), "logit": float(logit)}
                for token_id, logit in zip(top_ids.tolist(), top_values.tolist(), strict=True)
            ],
        })
        if policy["generation"]["stopOnEos"] and next_token in eos_set:
            break
        next_input = torch.tensor([[next_token]], dtype=torch.long)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)

    prompt_ids = [int(item) for item in input_ids[0].tolist()]
    output = {
        "schema": "doppler.pinned-source-transcript/v1",
        "model": policy["model"],
        "revision": policy["revision"],
        "entryPoint": "text.generate",
        "prompt": policy["prompt"],
        "promptTokenIds": prompt_ids,
        "generatedTokenIds": generated_ids,
        "generatedTokens": len(generated_ids),
        "generationSteps": generation_steps,
        "generation": policy["generation"],
        "execution": {
            "sampling": policy["generation"]["sampling"],
            "device": policy["execution"]["device"],
            "dtype": policy["execution"]["dtype"],
            "attentionImplementation": policy["execution"]["attentionImplementation"],
        },
        "identity": {
            "policySha256": "sha256:" + sha256_file(policy_path),
            "scriptSha256": "sha256:" + sha256_file(Path(__file__).resolve()),
            "promptSha256": "sha256:" + sha256_text(policy["prompt"]),
            "configSha256": "sha256:" + sha256_file(model_dir / "config.json"),
            "generationConfigSha256": "sha256:" + sha256_file(model_dir / "generation_config.json"),
            "tokenizerSha256": "sha256:" + sha256_file(model_dir / "tokenizer.json"),
            "sourceAcquisitionReceiptDigest": acquisition["receiptDigest"],
            "transformersCommit": observed_commit,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "loadEvidence": load_evidence,
        "boundaries": [
            {"boundaryId": name, **capture}
            for name, capture in captures.items()
        ],
        "decoded": {
            "withSpecialTokens": tokenizer.decode(generated_ids, skip_special_tokens=False),
            "withoutSpecialTokens": tokenizer.decode(generated_ids, skip_special_tokens=True),
        },
        "author": policy["author"],
    }
    output_path = (repo_root / policy["output"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
