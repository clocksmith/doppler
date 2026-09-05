#!/usr/bin/env python3
"""Retain independent, pinned HF source logits and tokens for a rerank contract."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def capture(policy):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if policy["schema"] != "doppler.rerank-source-capture/v1":
        raise ValueError("Unsupported capture schema")
    if policy["referenceDevice"] != "cpu" or policy["referenceDtype"] != "float32":
        raise ValueError("This reference lane requires explicit CPU float32 execution")
    source = Path(policy["sourceDirectory"])
    names = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "README.md"]
    for name in names:
        metadata = source / ".cache" / "huggingface" / "download" / (name + ".metadata")
        if metadata.read_text().splitlines()[0] != policy["revision"]:
            raise ValueError(f"Unpinned source file: {name}")
    manifest = json.loads(Path(policy["manifestPath"]).read_text())
    if manifest["modelId"] != policy["modelId"] or manifest["artifactIdentity"]["sourceCheckpointId"] != policy["repository"]:
        raise ValueError("Source and manifest identities differ")
    config = manifest["inference"]["rerank"]
    if config["format"] != "qwen3_yes_no_logit" or config["probability"] != "sigmoid":
        raise ValueError("Unsupported reranking contract")
    if config["score"] not in ["true_logit", "logit_difference"]:
        raise ValueError("Unsupported score policy")
    torch.set_num_threads(policy["threads"])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                               dtype=torch.float32, attn_implementation="eager").eval()
    outputs = []
    for index, document in enumerate(policy["input"]["documents"]):
        body = config["inputTemplate"]
        for key, value in {"instruction": config["instruction"], "query": policy["input"]["query"], "document": document}.items():
            body = body.replace("{" + key + "}", value)
        prompt = config["prefix"] + body + config["suffix"]
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**encoded).logits[0, -1]
        yes, no = float(logits[config["trueTokenId"]]), float(logits[config["falseTokenId"]])
        score = yes if config["score"] == "true_logit" else yes - no
        outputs.append({"index": index, "document": document, "tokenIds": encoded.input_ids[0].tolist(),
                        "trueLogit": yes, "falseLogit": no, "score": score, "probability": 1 / (1 + math.exp(-score))})
    return {
        "schema": "doppler.rerank-source-reference/v1",
        "source": {"checkpointId": policy["repository"], "repository": policy["repository"], "revision": policy["revision"],
                   "engine": "hf-transformers-pytorch", "torchVersion": torch.__version__, "transformersVersion": transformers.__version__,
                   "pythonVersion": platform.python_version(), "device": "cpu", "dtype": "float32", "attention": "eager",
                   "files": [{"path": name, "hash": digest(source / name)} for name in names]},
        "input": policy["input"], "scoringConfig": config, "tolerances": policy["tolerances"], "outputs": outputs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise ValueError("Reference output already exists; retain prior observations")
    policy = json.loads(Path(args.policy).read_text())
    result = capture(policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest(output), "documents": len(result["outputs"])}))
