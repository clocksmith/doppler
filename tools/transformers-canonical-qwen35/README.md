# Canonical Qwen3.5 export investigation

Internal, untimed tools for the exact 321-tensor canonical F32 text checkpoint.
This directory does not currently provide a full-model Transformers.js runner.
It never downloads model weights or substitutes a published ONNX checkpoint.

## Established scope

- `binding.py` and `inspect_canonical.py` verify canonical inventory/bytes and map
  every tensor to the pinned native HF text class. Input embeddings and output
  head remain separate. Norm offsets and raw `A_log` semantics are preserved.
- `export_layer.py` lowers the canonical linear-attention layer into explicit
  state inputs/outputs. Prefill uses native chunk64 mathematics, with a declared
  maximum of 128 tokens and zero-padded extra chunks; decode uses the native
  one-token recurrent formula. It does not force token-at-a-time prefill.
- `validate_layer.py` compares the adapter and exported ONNX against unchanged
  native HF on CPU using the exact canonical layer 0 parameters. Cases include
  lengths 1, 2, 28, 63, 64, 65, 127, 128, nonzero incoming decode states, and four
  prefill/decode partitions. Tolerances are frozen in `contract.json`.
- `onnx_lower.py` decomposes the export optimizer's RMSNormalization fusion
  back into the native F32 arithmetic supported by the pinned WebGPU operator
  implementation. No numerical CPU fallback is introduced.
- `run_layer_gpu.js` and `layer_gpu_browser.js` admit those two layer graphs
  with only WebGPU and `session.disable_cpu_ep_fallback=1`; state outputs must
  actually reside in GPU buffers. The retained run passed all 60 checks on the
  AMD rdna-3 adapter. This is single-layer evidence, not full-model support.
- `reference_native.py` runs the complete native HF text model on CPU. Its
  explicit F16 lane physically stores K/V as F16 and widens the stored values
  for unchanged eager F32 attention; convolution/recurrent state stays F32.
  Its F32 option is a separately labeled diagnostic lane.

`evidence/` retains exact binding, layer CPU/ONNX/GPU receipts, provider/source
identity, and a full-model native reference summary. Large ONNX graphs, binary
fixtures, full logits and model weights remain in the Jupiter paths bound by
those receipts. Saved results establish only their recorded scope and bytes.

## Running

Read `contract.json` for pinned dependencies. Use an isolated environment; the
existing Jupiter task environment is
`/home/x/.local/state/doppler/transformers-canonical-qwen35-20261029/export-venv`.
All Python numerical tools are CPU-only. A browser GPU run requires the shared
GPU slot. The runner takes one JSON config containing `modelDir`, `ortDist`,
`outputDir`, `browserExecutable`, `browserArgs`, `requiredVendor`, and `timeoutMs`.
The ORT distribution must be the pinned dependency of the inspected TJS install.

```sh
python -m unittest discover -s tools/transformers-canonical-qwen35 -p test_binding.py
python tools/transformers-canonical-qwen35/inspect_canonical.py --canonical PATH --out NEW_FILE --verify-bytes
python tools/transformers-canonical-qwen35/validate_layer.py --canonical PATH --out NEW_DIRECTORY --export
node tools/transformers-canonical-qwen35/run_layer_gpu.js CONFIG_JSON
```

The full-model graph, 48 cache bindings, TJS public API integration, initializer
ledger, full-model parity and performance comparison remain required work.
