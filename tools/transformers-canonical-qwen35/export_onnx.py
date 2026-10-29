"""Export full canonical text sessions after the independent native CPU gate."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import faulthandler
import json
from pathlib import Path
import shutil
import sys
import traceback

from binding import digest, read_inputs, verify_bytes
from native_model import load_native_model


def main():
    sys.stdout.reconfigure(line_buffering=True)
    faulthandler.dump_traceback_later(45, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--cpu-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    graph_dir = args.out / "onnx"
    graph_dir.mkdir()
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    lane = policy["textExport"]
    inputs = read_inputs(args.canonical, policy)
    verify_bytes(inputs, policy["hashChunkBytes"])
    proof = json.loads(args.cpu_receipt.read_text())
    if (proof.get("schema") != "doppler.canonical-text-export-cpu.v1" or proof.get("completed") is not True
            or proof.get("inputs") != inputs["inputs"] or not all(c["pass"] for c in proof["checks"])):
        raise ValueError("full-model CPU correctness gate is incomplete or mismatched")
    for filename, checksum in proof["sourceSha256"].items():
        if digest(Path(__file__).with_name(filename)) != checksum:
            raise ValueError("CPU-validated frontend source changed: " + filename)
    if proof["policySha256"] != digest(policy_path):
        raise ValueError("CPU-validated export policy changed")
    expected = policy["frameworks"] | policy["exportFrontend"]["frameworks"]
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise ValueError("ONNX export dependency identity mismatch")
    import torch
    import onnx
    from export_text import TextDecoderExport, EmbeddingExport, empty_cache, cache_names
    from onnx_lower import lower_rms_normalization
    from initializer_binding import bind_initializers
    from compile_optimizer import optimize

    torch.set_num_threads(policy["threads"])
    torch.set_num_interop_threads(policy["threads"])
    model, config, mappings = load_native_model(inputs, policy)
    receipt = {"schema": "doppler.canonical-text-onnx-export.v1", "completed": False, "inputs": inputs["inputs"],
               "scope": "Canonical ONNX construction and initializer binding; no ONNX inference, TJS, GPU or speed proof",
               "frameworks": versions, "parameterCount": len(mappings), "policySha256": digest(policy_path),
               "cpuGate": {"path": str(args.cpu_receipt), "sha256": digest(args.cpu_receipt)},
               "contract": lane, "graphs": [], "cacheBindings": [{"input": a, "output": b}
                                                                     for a, b in cache_names(config)]}
    source_dir = args.out / "sources"
    source_dir.mkdir()
    receipt["sourceFiles"] = {}
    for filename in ("export_onnx.py", "export_text.py", "export_layer.py", "onnx_lower.py", "initializer_binding.py",
                     "native_model.py", "binding.py", "contract.json", "compile_optimizer.py", "export-compiler.json"):
        path = Path(__file__).with_name(filename)
        shutil.copyfile(path, source_dir / filename)
        receipt["sourceFiles"][filename] = digest(path)
    graph_paths = {}
    try:
        with torch.no_grad():
            dimension = torch.export.Dim("sequence_length", min=2, max=lane["prefillBound"])
            embedding = torch.export.export(EmbeddingExport(model).eval(), (torch.zeros(1, 2, dtype=torch.int64),),
                                            dynamic_shapes=({1: dimension},), strict=True)
            program = torch.onnx.export(embedding, dynamo=True, optimize=False, input_names=["input_ids"],
                                       output_names=["inputs_embeds"], opset_version=policy["exportFrontend"]["opset"])
            optimize(program)
            path = graph_dir / "embed_tokens.onnx"
            program.save(path, external_data=True)
            graph_paths["embed_tokens"] = path
            del program, embedding
            for mode in ("prefill", "decode"):
                print(json.dumps({"graph": mode, "phase": "torch-capture"}), flush=True)
                # Example1 is specialized by torch.export; example2 keeps the
                # explicitly declared past-length range1..2047 symbolic.
                length, past_length = (2, 0) if mode == "prefill" else (1, 2)
                cache = empty_cache(config, past_length)
                arguments = (torch.zeros(1, length, config.hidden_size), torch.zeros(3, 1, length, dtype=torch.int64),
                             torch.ones(1, length + past_length, dtype=torch.int64), cache)
                if mode == "prefill":
                    dynamic = ({1: dimension}, {2: dimension}, {1: dimension}, tuple({} for t in cache))
                else:
                    past = torch.export.Dim("past_sequence_length", min=1, max=lane["maxSequenceLength"] - 1)
                    states = tuple({2: past} if tensor.dtype == torch.float16 else {} for tensor in cache)
                    dynamic = ({}, {}, {1: past + 1}, states)
                exported = torch.export.export(TextDecoderExport(model, lane, mode).eval(), arguments,
                                               dynamic_shapes=dynamic, strict=True)
                print(json.dumps({"graph": mode, "phase": "onnx-translation"}), flush=True)
                program = torch.onnx.export(exported, dynamo=True, optimize=False,
                    input_names=["inputs_embeds", "position_ids", "attention_mask"] + [p for p, n in cache_names(config)],
                    output_names=["logits"] + [n for p, n in cache_names(config)],
                    opset_version=policy["exportFrontend"]["opset"])
                print(json.dumps({"graph": mode, "phase": "explicit-compiler-passes"}), flush=True)
                optimize(program)
                path = graph_dir / ("decoder_" + mode + ".onnx")
                program.save(path, external_data=True)
                graph = onnx.load(path, load_external_data=False)
                lowerings = lower_rms_normalization(graph, check=False)
                onnx.save(graph, path)
                onnx.checker.check_model(str(path))
                graph_paths["decoder_" + mode] = path
                receipt["graphs"].append({"mode": mode, "path": str(path), "sha256": digest(path), "lowerings": lowerings,
                    "operators": dict(sorted(Counter(n.op_type for n in graph.graph.node).items())),
                    "inputs": [v.name for v in graph.graph.input], "outputs": [v.name for v in graph.graph.output]})
                print(json.dumps({"graph": mode, "export": "passed"}), flush=True)
                del program, exported, graph, cache, arguments
            initializer_proof = bind_initializers(inputs, policy, model, graph_paths)
            (args.out / "initializer-binding.json").write_text(json.dumps(initializer_proof, indent=2) + "\n")
            receipt["initializerBinding"] = {"path": str(args.out / "initializer-binding.json"),
                                                "sha256": digest(args.out / "initializer-binding.json"), "pass": True}
        receipt["artifacts"] = [{"path": str(p.relative_to(args.out)), "bytes": p.stat().st_size, "sha256": digest(p)}
                                for p in sorted(graph_dir.iterdir()) if p.is_file()]
        shutil.copyfile(args.canonical / "config.json", args.out / "config.json")
        shutil.copyfile(args.canonical / "tokenizer.json", args.out / "tokenizer.json")
        receipt["completed"] = True
    except Exception as error:
        receipt["error"] = {"exception": type(error).__name__, "message": str(error)}
        (args.out / "failure.log").write_text(traceback.format_exc())
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    faulthandler.cancel_dump_traceback_later()
    if not receipt["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
