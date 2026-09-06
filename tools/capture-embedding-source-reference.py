#!/usr/bin/env python3
"""Capture pinned Qwen source vectors on CPU, independently of Doppler execution."""
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
    if policy["repository"] != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError("This source oracle implements the Qwen last-token contract only")
    if policy["referenceDevice"] != "cpu" or policy["referenceDtype"] != "float32":
        raise ValueError("This reference requires CPU float32 execution")
    contract = policy["embeddingContract"]
    if contract["postprocessor"] != {"poolingMode": "last", "includePrompt": True,
                                    "projections": [], "normalize": "l2"}:
        raise ValueError("Unsupported source embedding semantics")
    source = Path(policy["sourceDirectory"])
    names = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "README.md"]
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
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state[:, -1]
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
