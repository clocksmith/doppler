# Benchmark Methodology

Canonical policy for benchmark fairness and claim publication.

## Claim requirements

Every claim must include:
- workload id
- engine/version
- cache/load mode
- command used
- artifact path(s)

## Canonical timing contract

Required metrics:
- `modelLoadMs`
- `firstTokenMs`
- `firstResponseMs`
- `prefillMs`
- `decodeMs`
- `totalRunMs`
- `decodeTokensPerSec`
- `prefillTokensPerSec`
- `decodeMsPerTokenP50`
- `decodeMsPerTokenP95`
- `decodeMsPerTokenP99`

Compare-artifact note:
- Cross-engine compare artifacts use apples-to-apples prompt metrics only:
  `firstTokenMs` and `promptTokensPerSecToFirstToken`.
- Raw engine payloads may still include `prefillMs` and `prefillTokensPerSec`, but those stay engine-local unless the semantics are proven identical.
- Capability matrices and release-matrix compare summaries must be updated when the compare metric contract changes.

## Fair comparison rules

- same model identity and quantization class, or explicit equivalence map
- same prompt/workload settings
- same sampling settings
- same hardware class/browser family for direct claims
- same explicit load-mode contract for the compared lane
- explicit disclosure of any deviation

Product-engine note:
- Best-available product comparisons may compare different artifact formats when
  the artifact formats are the promoted path for each engine.
- The receipt must state the format for each side, for example Doppler
  RDRR/direct-source LiteRT vs Transformers.js ONNX/q4f16.
- Do not describe a product-engine comparison as a format-identical kernel
  comparison.
- If exact greedy output parity is not required for a product-format throughput
  lane, the checked-in compare profile must declare `outputParityPolicy` with
  `requireMatch=false` and `matchMode="decode-valid"`. The artifact remains a
  throughput receipt only; it must not be cited as correctness-parity evidence.
- If one engine has no supported format lane, record that as a capability gap
  and keep the paired speed claim scoped to the best available supported path.

LiteRT/TFLite note:
- Doppler direct-source `.tflite`, `.task`, and `.litertlm` support is an
  experimental capability lane until promoted by a saved compare artifact and
  the support matrix.
- Transformers.js compare lanes currently support `onnx` and `safetensors`.
  LiteRT.js is a separate Google runtime, not a Transformers.js format lane.
- A Doppler LiteRT result can support a product claim only when the artifact
  proves browser execution, prompt parity, non-zero decode tokens, sane output,
  and an explicit disclosure that Transformers.js is using its best available
  ONNX/WebGPU path.

Prompt-target note:
- Shared `prefillTokens` workload targets refer to actual model-input prompt tokens.
- When compare runners synthesize a prompt from `prefillTokens`, they must resolve it with the selected tokenizer first.
- If either engine reports a different prompt-token count than the shared target, or the engines disagree on prompt-token count, the paired section is invalid.

## Near-parity statistical protocol

Use this protocol when a remaining decode difference may be smaller than normal
GPU clock, thermal, browser, or scheduling variation:

1. Interleave engine samples as adjacent pairs. Do not run every Doppler sample
   and then every challenger sample. Alternate which engine runs first when the
   harness supports it, and record the order.
2. Collect at least 20 valid pairs. Use 512 to 1024 decode tokens per sample when
   the model context allows it so fixed overhead and clock drift are a smaller
   fraction of the measurement.
3. Keep the artifact, prompt tokens, sampling tuple, load mode, decode cadence,
   browser, hardware, and power state fixed.
4. Require the configured output-parity gate on every pair. A faster mismatched
   output is not a timing sample for the parity population.
5. Convert each engine's rate to per-token decode time and calculate the paired
   difference for each adjacent pair.
6. Report the median and minimum per-token decode time for each engine, plus the
   paired 95% confidence interval for the mean difference. The minimum is a
   useful least-interrupted lower bound, but it does not replace the paired
   interval or the median.

Interpret a difference as `Doppler ms/token - challenger ms/token`:

- an interval entirely below zero supports a Doppler decode win
- an interval entirely above zero supports a challenger decode win
- an interval crossing zero with an absolute median throughput difference below
  0.5% is parity within measurement noise; stop tuning that lane
- an interval crossing zero with a larger observed difference is inconclusive;
  collect more evidence or isolate an uncontrolled factor

Do not publish a point estimate as a win when this stopping rule says parity or
inconclusive. Keep the raw paired samples and the summary artifact together.

## Compare runner contract

- Compare load mode must come from one explicit source:
  `--load-mode`, or `benchmarks/vendors/compare-engines.config.json defaults`.
- Warm and cold defaults are config-owned. Runner code must not infer load mode from `cacheMode`.
- Model-scoped Doppler tuning for compare lanes must be declared explicitly in
  `benchmarks/vendors/compare-engines.config.json`, typically through
  `dopplerRuntimeProfileByDecodeProfile`.
- Model-scoped output-parity policy must be declared in
  `benchmarks/vendors/compare-engines.config.json`. Runner code must not infer a
  relaxed parity rule from model names or prose.
- Compare-managed prompt, sampling, and decode-cadence fields must override any
  model-scoped Doppler runtime profile values so the reported lane contract stays stable.
- Doppler compare artifacts must distinguish request-level cadence from resolved
  execution-plan cadence. `runtimeMirror` records the runtime overlay, while the
  top-level cadence fields and `executionPlan` record what actually executed.
- Canonical Transformers.js compare repo/dtype defaults come from `models/catalog.json` `vendorBenchmark.transformersjs`, not ad hoc runner fallbacks.
- Benchmark-selected model registry lanes must cite their checked-in runtime
  report, compare result, and SVG summary through `models/catalog.json`
  `benchmarkEvidence`.
- Warm `opfs` lanes must fully prime the lazy generation path before the timed pass. A load-only warmup is not a valid warm-load comparison for models that fetch decoder assets on first token.
- Browser-side claim lanes must not depend on live remote asset fetches when a staged local snapshot is required for stability. When the TJS model is staged locally, the compare contract should use `--tjs-local-model-path` so warm comparisons are reproducible.
- A compare lane is claim-valid only when both engines finish under the same resolved contract.
- If one engine fails, the artifact must mark the section as an invalid paired comparison instead of presenting it as a speed result.
- Compare runners must not retry with different semantics inside the same reported lane.

## Tooling contract

Registry and runner behavior are canonical in
[../benchmarks/vendors/README.md](../benchmarks/vendors/README.md).
Implementation conventions are in
[style/benchmark-style-guide.md](style/benchmark-style-guide.md).
