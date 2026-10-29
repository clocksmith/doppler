"""CPU-only native HF linear-attention ONNX frontend probe; synthetic weights.

This is a structural probe, never an equal-weight or numerical parity receipt.
It uses canonical geometry and the installed native HF implementation unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import traceback

from binding import digest, read_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    policy_path = Path(__file__).with_name("contract.json")
    policy = json.loads(policy_path.read_text())
    inputs = read_inputs(args.canonical, policy)
    frontend = policy["exportFrontend"]
    expected = {**policy["frameworks"], **frontend["frameworks"]}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise ValueError("frontend dependency version mismatch: " + repr(versions))

    import torch
    import onnx
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    if hf.is_fast_path_available or hf.FusedRMSNormGated is not None:
        raise ValueError("probe requires native CPU implementation without CUDA hooks")
    torch.set_num_threads(policy["threads"])
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(**inputs["config"]["text_config"])
    layer = hf.Qwen3_5GatedDeltaNet(config, 0).eval().to("cpu")

    class ProbeLayer:
        def __init__(self, conv, recurrent):
            self.conv_states = conv
            self.recurrent_states = recurrent

    class ProbeCache:
        def __init__(self, conv, recurrent, previous):
            self.layers = [ProbeLayer(conv, recurrent)]
            self.previous = previous

        def has_previous_state(self, index):
            return self.previous

        def update_conv_state(self, value, index):
            self.layers[index].conv_states = value
            return value

        def update_recurrent_state(self, value, index):
            self.layers[index].recurrent_states = value
            return value

    class Probe(torch.nn.Module):
        def __init__(self, previous):
            super().__init__()
            self.layer = layer
            self.cache = ProbeCache(None, None, previous)

        def forward(self, hidden, conv, recurrent):
            cache = self.cache
            cache.layers[0].conv_states = conv
            cache.layers[0].recurrent_states = recurrent
            output = self.layer(hidden, cache_params=cache)
            return output, cache.layers[0].conv_states, cache.layers[0].recurrent_states

    conv = torch.zeros(1, layer.conv_dim, layer.conv_kernel_size)
    recurrent = torch.zeros(1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
    receipt = {"schema": "doppler.transformers-qwen35-frontend-probe.v1",
               "scope": "CPU structural export of native HF layer with synthetic parameters; no canonical ONNX or parity proof",
               "inputs": inputs["inputs"], "frameworks": versions,
               "nativeSourceSha256": digest(Path(hf.__file__)),
               "geometry": {"hidden": config.hidden_size, "conv": list(conv.shape),
                            "recurrent": list(recurrent.shape)}, "attempts": []}
    cases = [("decode-fixed", True, 1, None), ("prefill-fixed", False, 2, None)]
    cases.extend((f"prefill-dynamic-{maximum}", False, frontend["dynamicSequenceMin"], maximum)
                 for maximum in frontend["dynamicSequenceMaxima"])
    for name, previous, length, maximum in cases:
        dynamic = ({"hidden": {1: torch.export.Dim("sequence_length", min=length, max=maximum)},
                    "conv": {}, "recurrent": {}} if maximum else None)
        log = io.StringIO()
        attempt = {"name": name, "previousState": previous, "exampleSequenceLength": length,
                   "requestedSequenceRange": [length, maximum] if maximum else [length, length]}
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), torch.no_grad():
                model = Probe(previous).eval()
                arguments = (torch.zeros(1, length, config.hidden_size), conv.clone(), recurrent.clone())
                exported = torch.export.export(model, arguments, dynamic_shapes=dynamic, strict=True)
                program = torch.onnx.export(
                    exported, dynamo=True, input_names=["hidden", "conv", "recurrent"],
                    output_names=["output", "present_conv", "present_recurrent"],
                    opset_version=frontend["opset"], optimize=False)
                program.save(args.out / (name + ".onnx"))
            shape = [d if isinstance(d, int) else str(d) for d in program.model.graph.inputs[0].shape]
            if dynamic and isinstance(program.model.graph.inputs[0].shape[1], int):
                raise ValueError("dynamic sequence contract silently specialized to " + repr(shape))
            graph = onnx.load(args.out / (name + ".onnx"), load_external_data=False)
            onnx.checker.check_model(graph)
            attempt.update({"export": "passed", "hiddenShape": shape,
                            "rangeConstraints": str(exported.range_constraints),
                            "operators": dict(sorted(Counter(n.op_type for n in graph.graph.node).items())),
                            "graphSha256": digest(args.out / (name + ".onnx")),
                            "inputs": [v.name for v in program.model.graph.inputs],
                            "outputs": [v.name for v in program.model.graph.outputs]})
        except Exception as error:
            attempt.update({"export": "failed", "exception": type(error).__name__, "message": str(error)})
            traceback.print_exc(file=log)
        (args.out / (name + ".log")).write_text(log.getvalue())
        receipt["attempts"].append(attempt)
        print(json.dumps({"attempt": name, "export": attempt["export"]}), flush=True)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
