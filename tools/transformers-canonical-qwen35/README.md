# Canonical Qwen3.5 export investigation

Internal, untimed tools for the exact 321-tensor canonical F32 text checkpoint.
The full-model Transformers.js runner is diagnostic tooling under validation.
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
- `teacher_native.py` generated 64 greedy tokens for each of the four frozen
  prompts (28, 31, 34, and 655 input tokens). It retains all 248320 logits at
  steps 0, 1, 2, 4, 8, 16, 32, and 63, exact token IDs/text, EOS handling, and
  physical cache checks at every step. None of these runs encountered EOS.
- `export_text.py` and `validate_text.py` compose all 24 layers with the 48
  explicit cache ports and F16 physical full-attention K/V. The CPU adapter is
  bit-exact against unchanged native HF across 735 logits/state comparisons:
  prefill lengths 1, 2, and 28, each followed by four stateful decode steps.
  `export_onnx.py` constructed embedding, prefill and decode ONNX sessions and
  bound all 321 learned parameters to reachable initializer bytes/shapes.
- F16-KV ONNX CPU execution failed the frozen gate: four of 15 logit vectors
  exceeded 0.001, and strict internal state checks failed. The separate F32-KV
  variant passed all 15 logit checks (maximum error 0.0000537), while 53 internal
  state checks still failed their unchanged tighter tolerances. Both failures
  remain retained; neither artifact is qualified for timing.
- `diagnose_state.py` replayed the first F32 state failure at layer 16. State
  binding reconstructed exactly; native normalization on the same ONNX input
  was bit-exact. An inherited hidden difference and a local F32 matrix multiply
  difference account for the failed state comparison. This does not waive it.
- `teacher-f32kv.json` selects the separately named all-F32-KV teacher and binds
  the parent's separate lane contract. All four prompts reached 64 tokens and
  all eight full-logit snapshots. The original F16 teacher bytes are unchanged.
- `variant_f32kv.py` changes only cache casts/IO metadata, verifies learned
  initializer protobufs remain byte-identical, and shares single-linked external
  payloads through sibling graph files. Session GPU weight residency can still
  be duplicated and must be reported separately from on-disk payload sharing.
- `tjs_model.js` supplies explicit session routing through the public TJS model
  constructor. `run_text_gpu.js` uses public TJS embedding, forward and cache
  APIs. Its only supported mode is an untimed diagnostic that discloses the
  failed strict CPU state gate; it has no timing-admission path.
- The retained full-model AMD WebGPU diagnostic completed all 15 cases through
  those public TJS APIs, with CPU fallback explicitly disabled. Every full
  vocabulary row passed the frozen 0.001 logit tolerance (maximum 0.000126362),
  with 15 exact argmax matches. All 48 cache outputs in every case were actual
  F32 GPU buffers. GPU and browser error lists were empty. This establishes
  execution and output-logit accuracy for this custom F32-KV adapter, not strict
  internal-state qualification, four-prompt/64-token qualification, platform
  performance, or GPU readback of all learned weights.
- `audit_graph_ops.py` and `probe_ops.js` isolate exact operator/type/rank
  admission under the pinned runtime. Of 788 signatures in the host-geometry
  candidate, 786 admitted; runtime registry logs rejected F32 `Cos` and `Sin`.
  The package's TypeScript sources listing these operators did not establish
  registration in its shipped WASM runtime.
- The executable candidate uses three explicit export transformations:
  `variant_gpu_indices.py` keeps bounded application indices I32 and replaces
  exact squares with multiplication; `variant_host_shapes.py` declares integer
  allocation geometry from input dimensions; `variant_rotary_lookup.py` builds
  bounded rotary tables at export time using the source ONNX CPU subgraph.
  Runtime rotary evaluation uses WebGPU Gather/Where operations. It accepts
  positions 0 through 2047 and rejects out-of-bound application positions.
  These transformations preserve all 735 CPU comparison metrics and all saved
  ONNX/native logit bytes exactly, including the original 53 strict failures.
  Learned initializer protobufs remain unchanged. No model tensor arithmetic
  moves to CPU at runtime.
- `export_bucket.py` constructs a separate 64-token prefill graph and binds all
  321 parameters while sharing the frozen external payloads. The native CPU
  adapter passed 1225 checks bit-exactly at lengths 1, 2, 28, 63 and 64, each
  followed by four decode steps. Its F32 ONNX variant passed all 25 logit rows
  (maximum error 0.0000537); 61 strict state checks failed. The original
  53-failure receipt remains separate and unchanged.
- `combine_buckets.py` and `bucketed_prefill.js` prepare explicit 64/704 bucket
  routing. The three short frozen prompts select one native chunk; the
  655-token prompt selects 11 chunks. CPU construction and focused routing
  tests passed; the composed bucket bridge has not yet passed GPU execution.
- `teacher_cases.js` prepares genuine four-prompt greedy checks using engine
  outputs as subsequent inputs, with both frozen EOS IDs. It does not feed
  reference tokens into the model after divergence. The retained output-only
  AMD run passed all 256 tokens exactly and all 32 full-vocabulary snapshots
  (maximum error 0.0000553131). All 12288 actual cache-output checks were F32
  GPU buffers. The actual ORT device was verified and had no GPU/browser
  errors. This used the original 704-bound graph, with its padding disclosed;
  it does not qualify the bucket bridge, strict state values or weight readback.
- `teacher_native.py --save-states` captured all 48 native F32 cache values at
  all 32 snapshot positions. Its 256 tokens and 32 logit hashes match the
  existing F32 teacher exactly. These new state files are bounded observations;
  the earlier teacher bytes and source snapshots remain unchanged.
- `materialization_graphs.py` and `observed_session.js` prepare initializer
  observation outputs in the actual inference sessions. The original
  inference nodes and initializer protobufs remain unchanged. This is
  scaffolding: the first GPU observer used a different device from ORT and
  failed validation; the corrected-device attempt stalled before token 0 and
  was stopped. Neither attempt proves a model-weight mismatch or successful
  weight readback. Both failures are retained.

The pinned native ORT EP creates its own device despite the requested device
setting. The runner now verifies adapter identity and attaches error/loss
listeners to the actual public `await ort.env.webgpu.device` result. The
earlier 15-row receipt listened on the requested device, so its empty GPU-error
list has that narrower scope. Direct readback transport is still under repair.

The original full-text frontend declares a 704-token prefill bound: its graph computes
11 padded chunks even for a 28-token prompt. This is extra work and a current
scheduler limitation. The prepared 64/704 bridge reports each selected bound
and padding explicitly, with four independent sessions and possible duplicate
GPU weight residency. Decode uses the separate recurrent graph. Any eventual
comparison must identify the custom TJS adapter, padding work, source graph,
and scheduler; these tools do not establish stock TJS performance.

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

Strict full-model numerical qualification, actual GPU readback for all learned
weights, strict GPU-state checks, GPU bucket checks and performance
comparison remain required work. Full-model
diagnostic configuration is retained on Jupiter at
`/home/x/.local/state/doppler/transformers-canonical-qwen35-20261029/tjs-full-gpu-diagnostic-06-config.json`;
its model and CPU receipt paths are frozen by `evidence/full-tjs-gpu-identity.json`.
