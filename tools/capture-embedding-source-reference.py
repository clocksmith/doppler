#!/usr/bin/env python3
"""Capture pinned source vectors on CPU, independently of Doppler execution."""
import argparse
import hashlib
import json
import platform
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def capture(policy):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if policy["schema"] != "doppler.embedding-source-capture/v1":
        raise ValueError("Unsupported capture policy")
    pooling = {"Qwen/Qwen3-Embedding-0.6B": "last",
               "sentence-transformers/all-MiniLM-L6-v2": "mean"}.get(policy["repository"])
    if pooling is None:
        raise ValueError("This source oracle requires an explicitly supported embedding contract")
    if policy["referenceDevice"] != "cpu" or policy["referenceDtype"] != "float32":
        raise ValueError("This reference requires CPU float32 execution")
    contract = policy["embeddingContract"]
    if contract["postprocessor"] != {"poolingMode": pooling, "includePrompt": True,
                                    "projections": [], "normalize": "l2"}:
        raise ValueError("Unsupported source embedding semantics")
    source = Path(policy["sourceDirectory"])
    names = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "README.md"]
    if pooling == "mean":
        names += ["1_Pooling/config.json", "modules.json", "sentence_bert_config.json"]
        config = json.loads((source / "1_Pooling/config.json").read_text())
        expected = {"word_embedding_dimension": 384, "pooling_mode_cls_token": False,
                    "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
                    "pooling_mode_mean_sqrt_len_tokens": False}
        if config != expected or contract["dimension"] != 384:
            raise ValueError("MiniLM source pooling configuration differs from the frozen contract")
        if policy.get("maxInputTokens") != 256:
            raise ValueError("MiniLM reference requires an explicit 256-token input bound")
        if json.loads((source / "sentence_bert_config.json").read_text())["max_seq_length"] != 256:
            raise ValueError("Source input bound differs from the reference policy")
        modules = json.loads((source / "modules.json").read_text())
        if [(item["idx"], item["type"]) for item in modules] != [
                (0, "sentence_transformers.models.Transformer"),
                (1, "sentence_transformers.models.Pooling"),
                (2, "sentence_transformers.models.Normalize")]:
            raise ValueError("Source module order differs from mean-pooling and L2 normalization")
    for name in names:
        metadata = source / ".cache" / "huggingface" / "download" / (name + ".metadata")
        if metadata.read_text().splitlines()[0] != policy["revision"]:
            raise ValueError(f"Unpinned source file: {name}")
    torch.set_num_threads(policy["threads"])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                     dtype=torch.float32, attn_implementation="eager").eval()
    outputs = []
    for index, text in enumerate(policy["input"]["texts"]):
        encoded = tokenizer(text, return_tensors="pt", truncation=False)
        if pooling == "mean" and encoded.input_ids.shape[1] > policy["maxInputTokens"]:
            raise ValueError("Reference input exceeds the declared token bound; truncation is forbidden")
        with torch.inference_mode():
            states = model(**encoded).last_hidden_state
            if pooling == "last":
                hidden = states[:, -1]
            else:
                mask = encoded.attention_mask.unsqueeze(-1).to(states.dtype)
                hidden = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
            vector = torch.nn.functional.normalize(hidden, p=2, dim=1)[0]
        if vector.numel() != contract["dimension"] or not torch.isfinite(vector).all():
            raise ValueError("Source vector violates the declared geometry")
        outputs.append({"text": text, "tokenIds": encoded.input_ids[0].tolist(), "embedding": vector.tolist()})
        print(f"Captured source vector {index + 1}/{len(policy['input']['texts'])}", flush=True)
    return {
        "schema": "doppler.embedding-source-reference/v1",
        "source": {"checkpointId": policy["repository"], "repository": policy["repository"],
                   "revision": policy["revision"], "engine": "hf-transformers-pytorch",
                   "torchVersion": torch.__version__, "transformersVersion": transformers.__version__,
                   "pythonVersion": platform.python_version(), "device": "cpu", "dtype": "float32",
                   "attention": "eager", "files": [{"path": name, "hash": digest(source / name)} for name in names]},
        "input": policy["input"], "embeddingContract": contract,
        "tolerances": policy["tolerances"], "outputs": outputs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise ValueError("Retain earlier observations; output already exists")
    result = capture(json.loads(Path(args.policy).read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(output), "hash": digest(output), "texts": len(result["outputs"])}))
