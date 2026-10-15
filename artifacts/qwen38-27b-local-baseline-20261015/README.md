# Qwen3.8-27B single-machine baseline, 2026-10-15

A conventional whole-model baseline runs on amd-host's Radeon 8060S using
llama.cpp Vulkan. This is not Doppler inference, Reploid partition qualification,
FP8 model execution, or product-demand evidence. The Mac is the browser client;
its 24 GiB physical memory does not hold this 29 GB Q8 artifact plus inference state.

## Identity and fit

- Source: `Qwen/Qwen3.8-27B`, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- All 18 source SafeTensors downloaded at that revision and checked against their
  published SHA-256 and sizes. Metadata hashes are retained in `host.json`.
- Converter/runtime: llama.cpp `59d5fad7c111d30a547d690b69235ed9e05c035d`.
- Conversion: `convert_hf_to_gguf.py source --outtype q8_0 --use-temp-file`.
  Weight matrices use Q8_0; selected small tensors remain F32. This is integer
  quantization, not FP8. No Q4 substitution or source-model change occurred.
- GGUF: 29,047,084,512 bytes; SHA-256
  `bb54d1417996e1222387f5ed5e2c8d234e390cfb97f4068c8f97e7153ccaf293`.
- Chat template: upstream tokenizer configuration's unmodified template, SHA-256
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- Text-only baseline; vision attachments and speculative/MTP execution are not
  exercised. The server explicitly reports unused MTP tensors.
- One slot, context 4,096, batch 256, microbatch 64, four CPU threads, 99 requested
  GPU layers, Jinja template supplied explicitly, thinking disabled, no warmup.
- Before loading, approximately 62.0 GB system memory was available. During the
  initial requests the sampled minimum was approximately 30.8 GB. Process RSS
  peaked around 2.0 GB; RSS does not account for all GPU allocations. No claim of
  complete peak GPU-memory accounting follows. The supervisor stops its server
  below 12 GiB system-available memory; this is not a guarantee against GPU OOM.
- Existing contributors were not stopped. These are functional timings under the
  host's existing workload, not an isolated or matched engine benchmark.

## Executed checks

- `smoke.js` / `smoke.json`: constrained fictional movie selection, a follow-up
  retaining its runtime, and arithmetic. All three complete correctly. The first
  answer has 59 completion tokens and completes in about 7.9 seconds on the repeat
  run; follow-up about 2.4 seconds; arithmetic about 0.75 seconds. Exact prompt,
  usage, prefill, decode and wall timings are retained.
- `browser.json` / `chat-answer.png`: Playwright opens the actual llama.cpp chat UI,
  submits a question through Send, and observes the answer `42`.
- `cancellation.json`: abort a streaming answer, then successfully ask another
  question on the same server. The replacement answer completes in under a second.
- `smoke-initial-mutable-metadata.json` retains an initial recording defect: the
  harness kept a reference to the messages array, so the first request's stored
  messages later included its follow-up. Outputs/timings are from that run, but its
  prompt metadata is superseded. The fixed harness snapshots each request and the
  complete three-check repeat is `smoke.json`.

Reproduce the HTTP checks against the running, loopback-bound server:

```sh
node artifacts/qwen38-27b-local-baseline-20261015/smoke.js \
  http://127.0.0.1:8127 /tmp/qwen27b-smoke.json
```

The local port is an SSH forward to the single AMD host, not a hosted API or
partitioned route. Endpoint availability depends on that server and tunnel.
The model's 262,144-token advertised context and general answer quality have not
been qualified by this bounded check. Q8 quality against source precision remains
unmeasured; the successful examples do not establish quantization equivalence.

## Packed eight-bit WebGPU experiment

`packed-8bit.js` / `packed-8bit-mac.json` executes on Apple Metal through Dawn:

- packs four bytes per `u32` storage word;
- decodes all 254 finite E4M3FN patterns to F32 and flags both nonfinite encodings;
- checks one FP8-packed/wider-arithmetic dot product against a CPU reference;
- checks signed `dot4I8Packed` against its exact integer result.

This proves the small shader mechanism on this device. It does not establish a
full-model FP8 representation, native FP8 multiplication, a performance gain,
scaling policy, or Doppler support. The initial probe's constant NaN construction
was rejected by WGSL validation; the final probe flags nonfinite encodings instead
of attempting to propagate NaNs. Runtime arithmetic uses wider values after FP8
unpacking. Neither inspected Mac nor AMD Dawn adapter advertised an FP8 feature;
packed integer dot product support is distinct from FP8 arithmetic.

```sh
node artifacts/qwen38-27b-local-baseline-20261015/packed-8bit.js \
  /tmp/packed-8bit-mac.json
```

Component: Doppler retained inference evidence. Intent: preserved. No runtime,
model catalog, production deployment, public product contract or Reploid package
was changed. This artifact records a conventional baseline and a bounded shader
probe; it does not promote either to Doppler model support.
