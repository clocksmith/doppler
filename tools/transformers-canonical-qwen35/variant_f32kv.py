"""Create a distinct F32-KV graph variant without changing learned initializers."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import onnx
from onnx import TensorProto

from binding import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads((args.source / "receipt.json").read_text())
    policy = json.loads(args.policy.read_text())
    if source.get("completed") is not True or source["contract"]["kvPhysicalDtype"] != "float16":
        raise ValueError("F32 variant requires a completed explicitly F16 artifact construction")
    if policy["kvPhysicalDtype"] != "float32":
        raise ValueError("variant policy must explicitly require physical F32 cache")
    allowed = {"kvPhysicalDtype", "scope"}
    if any(source["contract"][key] != value for key, value in policy.items() if key not in allowed):
        raise ValueError("F32 variant attempted to change an unrelated numerical/scheduler contract")
    args.out.mkdir(parents=True, exist_ok=False)
    payloads = []
    for artifact in source["artifacts"]:
        old = args.source / artifact["path"]
        if digest(old) != artifact["sha256"] or old.stat().st_size != artifact["bytes"]:
            raise ValueError("source ONNX artifact changed")
        if old.suffix == ".data":
            if old.stat().st_nlink != 1:
                raise ValueError("source external data must retain a single hard link for ONNX security checks")
            payloads.append(old)
    receipt = deepcopy(source)
    receipt["contract"] = policy
    receipt["scope"] = "Distinct F32-KV ONNX construction; no numerical, TJS, GPU or performance proof"
    receipt["sourceF16Artifact"] = {"path": str(args.source / "receipt.json"), "sha256": digest(args.source / "receipt.json")}
    receipt["sourceF16CpuAdapterGate"] = receipt.pop("cpuGate")
    receipt["cpuGate"] = {"status": "not-run", "scope": "F32 ONNX variant requires its own native-HF numerical gate"}
    receipt["sourceFiles"]["variant_f32kv.py"] = digest(Path(__file__))
    receipt["sourceFiles"]["text-export-f32kv.json"] = digest(args.policy)
    receipt["variantChanges"] = []
    receipt["sessionFiles"] = {"embed_tokens": str(args.source / "onnx" / "embed_tokens.onnx")}
    for graph in receipt["graphs"]:
        filename = Path(graph["path"]).name
        original = onnx.load(args.source / "onnx" / filename, load_external_data=False)
        model = deepcopy(original)
        cast_changes = []
        for node in model.graph.node:
            if node.op_type == "Cast" and not node.domain:
                target = next(attribute for attribute in node.attribute if attribute.name == "to")
                if target.i == TensorProto.FLOAT16:
                    target.i = TensorProto.FLOAT
                    cast_changes.append({"node": node.name, "output": list(node.output),
                                         "from": "float16", "to": "float32"})
        inputs = [v for v in model.graph.input if v.name.startswith("past_key_values.")]
        outputs = [v for v in model.graph.output if v.name.startswith("present.")]
        if len(cast_changes) != 12 or len(inputs) != 12 or len(outputs) != 12:
            raise ValueError("variant physical cache transform does not match six key/value pairs")
        for value in inputs + outputs:
            if value.type.tensor_type.elem_type != TensorProto.FLOAT16:
                raise ValueError("source cache IO was not physically F16")
        for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == TensorProto.FLOAT16:
                value.type.tensor_type.elem_type = TensorProto.FLOAT
        for old, new in zip(original.graph.initializer, model.graph.initializer):
            if old.SerializeToString() != new.SerializeToString() or new.data_type == TensorProto.FLOAT16:
                raise ValueError("variant changed a learned/auxiliary initializer or encountered a half initializer")
        if [(n.op_type, tuple(n.input), tuple(n.output)) for n in original.graph.node] != [
                (n.op_type, tuple(n.input), tuple(n.output)) for n in model.graph.node]:
            raise ValueError("variant changed computation graph connectivity")
        # ONNX forbids traversal references and hard-linked external payloads.
        # A sibling graph can safely reference the original single-linked data.
        target = args.source / "onnx" / (Path(filename).stem + "." + args.out.name + ".onnx")
        if target.exists():
            raise ValueError("refusing to overwrite an existing variant graph")
        onnx.save(model, target)
        onnx.checker.check_model(str(target))
        graph["path"], graph["sha256"] = str(target), digest(target)
        receipt["sessionFiles"]["decoder_" + graph["mode"]] = str(target)
        receipt["variantChanges"].append({"graph": graph["mode"], "cacheCasts": cast_changes,
            "inputTypes": [v.name for v in inputs], "outputTypes": [v.name for v in outputs],
            "initializerProtobufsByteUnchanged": True, "graphConnectivityUnchanged": True})
    old_binding = Path(source["initializerBinding"]["path"])
    if digest(old_binding) != source["initializerBinding"]["sha256"]:
        raise ValueError("source initializer binding receipt changed")
    shutil.copyfile(old_binding, args.out / "initializer-binding.json")
    receipt["initializerBinding"]["path"] = str(args.out / "initializer-binding.json")
    receipt["initializerBinding"]["unchangedFromF16"] = source["initializerBinding"]["sha256"]
    shutil.copytree(args.source / "sources", args.out / "sources")
    shutil.copyfile(Path(__file__), args.out / "sources" / "variant_f32kv.py")
    shutil.copyfile(args.policy, args.out / "sources" / "text-export-f32kv.json")
    for filename in ("config.json", "tokenizer.json"):
        shutil.copyfile(args.source / filename, args.out / filename)
    files = payloads + [Path(value) for value in receipt["sessionFiles"].values()]
    receipt["artifacts"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(files)]
    receipt["externalPayloadSharing"] = "Single-linked frozen source files; variant graph protobufs are siblings. Separate sessions may duplicate GPU residency."
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"variant": "physical F32 KV", "initializerCount": source["parameterCount"],
                      "cacheCastsPerDecoder": 12, "externalData": "single-linked frozen source files; sibling graph protobufs"}), flush=True)


if __name__ == "__main__":
    main()
