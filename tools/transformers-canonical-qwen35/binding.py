"""Canonical-byte and explicit name binding; no tensor arithmetic or inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_inputs(directory: Path, policy: dict) -> dict:
    manifest_path = directory / "canonical.json"
    config_path = directory / "config.json"
    weights = directory / "model.safetensors"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    if (manifest.get("schema") != policy["sourceSchema"] or
            manifest.get("completed") is not True or
            manifest.get("tensorPrefix") != policy["sourcePrefix"]):
        raise ValueError("canonical source identity/completion/prefix is unsupported")
    if (config.get("model_type") != policy["modelType"] or
            config.get("text_config", {}).get("model_type") != policy["textModelType"]):
        raise ValueError("configuration must explicitly describe the Qwen3.5 text backbone")
    if (config.get("tie_word_embeddings") is not False or
            config["text_config"].get("tie_word_embeddings") is not False):
        raise ValueError("canonical input and output matrices must remain untied")
    if config["text_config"].get("mtp_num_hidden_layers") != 0:
        raise ValueError("speculative/MTP layers are outside the text-only contract")
    with weights.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length header")
        length = struct.unpack("<Q", prefix)[0]
        if length > policy["headerLimitBytes"] or length > weights.stat().st_size - 8:
            raise ValueError("invalid or oversized safetensors header")
        encoded = stream.read(length)
    header = json.loads(encoded)
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    tensors = manifest["tensors"]
    names = [t["name"] for t in tensors]
    if len(names) != len(set(names)) or set(names) != set(entries):
        raise ValueError("canonical tensor inventory and safetensors header differ")
    spans = []
    for tensor in tensors:
        entry = entries[tensor["name"]]
        shape = tensor["shape"]
        if (not isinstance(shape, list) or not shape or
                any(type(d) is not int or d <= 0 for d in shape)):
            raise ValueError("tensor shape must have explicit positive dimensions")
        if tensor["dtype"] != "F32" or entry["dtype"] != "F32" or entry["shape"] != shape:
            raise ValueError("canonical F32 dtype/shape identity differs")
        begin, end = entry["data_offsets"]
        size = math.prod(shape) * 4
        if (type(begin) is not int or type(end) is not int or
                begin < 0 or end - begin != size or tensor["bytes"] != size):
            raise ValueError("tensor offsets/byte count differ from declared F32 shape")
        expected = tensor["canonicalBytesSha256"]
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("canonical tensor digest is not lowercase SHA-256")
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError("tensor payload spans overlap or leave an unbound gap")
        cursor = end
    if cursor != weights.stat().st_size - 8 - length:
        raise ValueError("unbound or truncated safetensors payload")
    return {"manifest": manifest, "config": config, "header": entries,
            "weights": weights, "payloadOffset": 8 + length,
            "inputs": {"canonicalManifestSha256": digest(manifest_path),
                       "configSha256": digest(config_path),
                       "safetensorsHeaderSha256": hashlib.sha256(prefix + encoded).hexdigest()}}


def verify_bytes(inputs: dict, chunk_size: int) -> None:
    with inputs["weights"].open("rb") as stream:
        for tensor in inputs["manifest"]["tensors"]:
            start, end = inputs["header"][tensor["name"]]["data_offsets"]
            stream.seek(inputs["payloadOffset"] + start)
            remaining = end - start
            checksum = hashlib.sha256()
            while remaining:
                chunk = stream.read(min(chunk_size, remaining))
                if not chunk:
                    raise ValueError("tensor payload truncated during verification")
                checksum.update(chunk)
                remaining -= len(chunk)
            if checksum.hexdigest() != tensor["canonicalBytesSha256"]:
                raise ValueError("canonical tensor byte hash mismatch: " + tensor["name"])


def bind_names(tensors: list[dict], shapes: dict[str, list[int]], policy: dict) -> list[dict]:
    mapped = []
    targets = set()
    for tensor in tensors:
        source = tensor["name"]
        target = policy["exactNames"].get(source)
        if target is None:
            if not source.startswith(policy["sourcePrefix"]):
                raise ValueError("tensor has no explicit target mapping: " + source)
            target = policy["targetPrefix"] + source[len(policy["sourcePrefix"]):]
        if target in targets or target not in shapes or shapes[target] != tensor["shape"]:
            raise ValueError("target identity/shape is missing, aliased or different: " + target)
        targets.add(target)
        mapped.append({"source": source, "target": target, "shape": tensor["shape"],
                       "dtype": "F32", "sha256": tensor["canonicalBytesSha256"],
                       "transform": "identity"})
    if targets != set(shapes):
        raise ValueError("target model has unbound parameters: " + repr(sorted(set(shapes) - targets)))
    return mapped
