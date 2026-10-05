"""Independent CPU decision logits: Transformers Gemma and GGUF Q4_K decoding.

Test-only reference. Never imported by Doppler Run. The evaluation contract fixes
model configuration, prompts, labels, numerical tolerance and task expectations.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from gguf.quants import Q4_K
from tokenizers import Tokenizer
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, Qwen3Config, Qwen3ForCausalLM


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_tensor(root, manifest, descriptor):
    spans = descriptor.get("spans") or [{"shardIndex": descriptor["shard"],
        "offset": descriptor["offset"], "size": descriptor["size"]}]
    chunks = []
    for span in spans:
        with (root / manifest["shards"][span["shardIndex"]]["filename"]).open("rb") as stream:
            stream.seek(span["offset"])
            chunk = stream.read(span["size"])
            if len(chunk) != span["size"]:
                raise ValueError("Truncated reference tensor")
            chunks.append(chunk)
    raw = b"".join(chunks)
    shape = descriptor["shape"]
    if descriptor["dtype"] == "Q4_K_M":
        value = Q4_K.dequantize_blocks(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 144))
        return torch.from_numpy(value.reshape(shape[0], -1)[:, :shape[1]].copy())
    if descriptor["dtype"] == "F16":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.float16).astype(np.float32)).reshape(shape)
    if descriptor["dtype"] == "BF16":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).float().reshape(shape)
    raise ValueError(f"Unsupported reference encoding: {descriptor['dtype']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract_bytes = args.contract.read_bytes()
    contract = json.loads(contract_bytes)
    manifest_bytes = (args.model / "manifest.json").read_bytes()
    if digest(manifest_bytes) != contract["manifestSha256"]:
        raise ValueError("Reference model manifest identity differs")
    manifest = json.loads(manifest_bytes)
    torch.set_num_threads(contract["reference"]["cpuThreads"])
    implementations = {"Gemma3ForCausalLM": (Gemma3TextConfig, Gemma3ForCausalLM),
                       "Qwen3ForCausalLM": (Qwen3Config, Qwen3ForCausalLM)}
    config_type, model_type = implementations[contract["reference"].get("implementation", "Gemma3ForCausalLM")]
    config = config_type(**contract["reference"]["modelConfig"])
    config._attn_implementation = "eager"
    model = model_type(config).float().eval()
    state = {name: read_tensor(args.model, manifest, descriptor)
             for name, descriptor in manifest["tensors"].items()}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    del state
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    rows = []
    with torch.inference_mode():
        for case in contract["cases"]:
            prompt = contract["prefix"] + case["question"] + contract["suffix"]
            tokens = tokenizer.encode(prompt).ids
            labels = []
            for choice in contract["choices"]:
                combined = tokenizer.encode(prompt + choice["label"]).ids
                if combined[:-1] != tokens or len(combined) != len(tokens) + 1:
                    raise ValueError("Reference label is not a single contextual token")
                labels.append(combined[-1])
            logits = model(torch.tensor([tokens]), use_cache=False, logits_to_keep=1).logits[0, -1].float()
            scores = [float(logits[token]) for token in labels]
            selected = contract["choices"][max(range(len(scores)), key=scores.__getitem__)]["id"]
            rows.append({"id": case["id"], "prompt": prompt, "promptTokenIds": tokens,
                         "tokenIds": labels, "logits": scores, "selectedId": selected,
                         "expectedId": case["expectedId"]})
    result = {"schema": "doppler.choice-scoring-cpu-reference/v1",
              "contractSha256": digest(contract_bytes), "modelManifestSha256": digest(manifest_bytes),
              "implementation": f"Transformers {model_type.__name__} eager Float32 CPU; GGUF Q4_K dequantization",
              "versions": {name: importlib.metadata.version(name)
                           for name in ["torch", "transformers", "gguf", "tokenizers", "numpy"]},
              "cases": rows, "correctChoices": sum(row["selectedId"] == row["expectedId"] for row in rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "cases": len(rows), "correctChoices": result["correctChoices"]}))


if __name__ == "__main__":
    main()
