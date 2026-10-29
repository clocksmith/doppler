"""Bind canonical parameters to actual reachable ONNX initializer bytes."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path

import onnx


def initializer_inventory(model_path):
    model = onnx.load(model_path, load_external_data=False)
    needed = {value.name for value in model.graph.output}
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            needed.update(node.input)
    inventory = []
    for tensor in model.graph.initializer:
        external = {entry.key: entry.value for entry in tensor.external_data}
        if external:
            location = Path(model_path).parent / external["location"]
            offset, length = int(external.get("offset", 0)), int(external["length"])
            checksum = hashlib.sha256()
            with location.open("rb") as stream:
                stream.seek(offset)
                remaining = length
                while remaining:
                    raw = stream.read(min(4 << 20, remaining))
                    if not raw:
                        raise ValueError("truncated initializer data: " + tensor.name)
                    checksum.update(raw)
                    remaining -= len(raw)
            data_hash = checksum.hexdigest()
        else:
            array = onnx.numpy_helper.to_array(tensor)
            length = array.nbytes
            data_hash = hashlib.sha256(array.tobytes()).hexdigest()
        inventory.append({"name": tensor.name, "dtype": tensor.data_type, "shape": list(tensor.dims),
                          "bytes": length, "sha256": data_hash, "reachable": tensor.name in needed,
                          "externalData": external or None})
    return inventory


def bind_initializers(inputs, policy, model, graph_paths):
    import torch

    inventory = {name: initializer_inventory(path) for name, path in graph_paths.items()}
    indexes = {}
    for graph_name, entries in inventory.items():
        index = defaultdict(list)
        for entry in entries:
            if entry["dtype"] == onnx.TensorProto.FLOAT and entry["reachable"]:
                if entry["bytes"] != math.prod(entry["shape"]) * 4:
                    raise ValueError("F32 initializer byte length disagrees with dimensions")
                index[(tuple(entry["shape"]), entry["sha256"])].append(entry["name"])
        indexes[graph_name] = index
    state = model.state_dict()
    parameters = []
    for tensor in inputs["manifest"]["tensors"]:
        source = tensor["name"]
        target = policy["exactNames"].get(source, policy["targetPrefix"] + source[len(policy["sourcePrefix"]):])
        value = state[target].detach()
        candidates = [("identity", tensor["shape"], tensor["canonicalBytesSha256"])]

        def prepared(name, array):
            contiguous = array.contiguous()
            candidates.append((name, list(contiguous.shape), hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()))

        if value.ndim == 2:
            prepared("transpose(1,0)", value.t())
        if target.endswith(".A_log"):
            prepared("negative-exp-f32", -value.float().exp())
        offset_norm = (target == "model.norm.weight" or
                       any(target.endswith(suffix) for suffix in (".input_layernorm.weight", ".post_attention_layernorm.weight",
                                                                  ".self_attn.q_norm.weight", ".self_attn.k_norm.weight")))
        if offset_norm:
            prepared("add-one-f32", value.float() + 1.0)
        graphs = ["embed_tokens"] if target == "model.embed_tokens.weight" else ["decoder_prefill", "decoder_decode"]
        bindings = []
        for graph in graphs:
            matches = []
            for transform, shape, checksum in candidates:
                for name in indexes[graph].get((tuple(shape), checksum), []):
                    matches.append({"graph": graph, "initializer": name, "transform": transform,
                                    "shape": shape, "sha256": checksum})
            if not matches:
                raise ValueError("canonical parameter has no byte/shape-bound reachable initializer: " + source + " in " + graph)
            bindings.extend(matches)
        parameters.append({"source": source, "target": target, "canonicalShape": tensor["shape"],
                           "canonicalBytesSha256": tensor["canonicalBytesSha256"], "bindings": bindings})
    head = next(p for p in parameters if p["target"] == "lm_head.weight")
    embedding = next(p for p in parameters if p["target"] == "model.embed_tokens.weight")
    if head["canonicalBytesSha256"] == embedding["canonicalBytesSha256"]:
        raise ValueError("canonical head and embedding identity unexpectedly coincide")
    return {"schema": "doppler.canonical-onnx-initializers.v1", "parameters": parameters,
            "parameterCount": len(parameters), "graphs": inventory,
            "headAndEmbedding": "separate canonical byte identities, separate session initializers", "pass": True}
