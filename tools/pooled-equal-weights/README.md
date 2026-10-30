# Common-weight engine comparison preparation

This directory prepares an explicit F32 comparison of Qwen3.5-0.8B text inference.
It does not establish a speed win or equivalent native quantized products.

1. `export-canonical.js <config.json>` verifies the pinned source manifest and
   every shard, then reconstructs selected learned tensors into F32 Safetensors.
   The output must be new. No source weights are rewritten or requantized.
2. Convert that directory with the canonical F32 conversion recipe through the
   existing Doppler `convert` command. `verify-rdrr.js <canonical> <rdrr> <report>`
   checks every converted tensor's shape, dtype, extent, and exact bytes.
3. `run-pooled-probe.js <config.json>` snapshots the supplied Pooled engine source,
   records its file hashes, and serves it on loopback. The adapter uses native F32
   kernels. It verifies all logical inputs and reads actual GPU storage, including
   fused views. Pooled's CPU embedding table is identified separately. Declared
   norm-offset and decay-coefficient preparation is recorded, never implicit.
4. `prepare-doppler-probe.js <config.json>` creates a separate untimed observation
   snapshot. Explicit `harnessRelativeDir` identifies the pinned source layout.
   With `captureLogitPositions`, it copies raw logits before sampling modifies
   them and requires complete, finite vocabulary rows at every requested
   position reached by the transcript. Optional single-token boundaries are
   captured after the request. Timings from this snapshot are inadmissible.
   Preparation publishes the final directory atomically only after every
   observation hook succeeds; failed preparations retain their own error record.
   Local model caches are excluded and dependencies are linked.
5. Compare prompt IDs, generated IDs, full numerical outputs, backend identities,
   and effective weight materializations before starting the timing contract.

An optional `timing` object on a Pooled probe config runs a resident timing pilot
after weight readback, instead of instrumented logits capture. It requires
`diagnosticOnly: true`, explicit `warmupRuns`/`timedRuns`, and pinned
`expectedTokenIds`. Every warmup and measured attempt is retained. The interval
includes prefill, greedy generation, and final queue completion; tokenization,
reset/drain, weight verification, and output checking are outside it. These
unpaired samples localize costs; they cannot establish the balanced-pair win.

The original F16-KV contract is
`benchmarks/vendors/pooled-equal-weights/contract.json`. The separate
`contract-f32kv.json` selects native Pooled F32 KV storage with
`attnFlash: false`, preserving the original numerical and performance thresholds.
Neither lane qualifies the other. Machine paths, browser
arguments, deadlines, and output directories are explicit run JSON fields;
retained examples are in the report archive. Never reuse an existing output.
The supplied vendor tree is read only, and its actual bytes—not an enclosing
repository's Git revision—are the source identity.

`prepare-pooled.js` can prepare an explicitly requested nearest-even control in
the copied vendor source. It checks pinned input hashes and records every
changed file; it never modifies the original engine. This control is distinct
from the native Pooled path and cannot silently replace it in timing evidence.

`rne-probe-browser.js` and `kv-cache-rne-prototype.wgsl` are an independent
rounding experiment. The prototype is not registered or shipped as a runtime
kernel. Both native and prototype conversion passed the retained Python
binary16 oracle; this experiment did **not** establish the cause of the model
mismatch. Optional prototype replacement is confined to an explicitly recorded
diagnostic snapshot and cannot supply performance evidence.

Focused checks: `node --test tools/pooled-equal-weights/*.test.js`.
Physical receipts and unresolved gates:
`reports/pooled-equal-weights-20261029/README.md`.

`run-resident-pairs.js prepare <config.json>` freezes committed sources and runs
CPU admission only. `run-resident-pairs.js run <prepared-output>` launches the
frozen committed runner; GPU execution requires a separately authorized slot.
The method is fixed before execution in `resident-method-f32kv.json`: four frozen
prompts, one warmup per engine/prompt, 20 alternating AB/BA pairs, no retries or
outlier removal, paired log-ratio geometric mean and fixed-prompt stratified
bootstrap (seed 0, 10,000 replicates, two-sided percentile 95% interval).
All thresholds come from the unchanged `contract-f32kv.json`.

The run config requires `outputDir` (new), `dopplerRoot`, full engine `dopplerCommit`, full current committed `harnessCommit`, `pooledRoot` (retained native byte snapshot), `pooledSourceIdentity` (`{ path, sha256 }`),
`canonicalDir`, `rdrrDir`, `ggufPath`, `browserExecutable`, `browserArgs`,
`requiredVendor`, `timeoutMs`, `dependencyPaths`, `referencesPath`, `proofBinding`,
and `doppler: { modelId, loadMode: "http", runtimeConfig }`. Dependencies are copied,
not linked; explicitly include `node_modules/playwright`, `node_modules/playwright-core`,
and any browser-loaded package assets. Both models remain resident in isolated
browser contexts of one frozen browser executable. Requests run serially.
Fresh actual-resource verification on both resident instances happens during
setup, outside both clocks. Source/dependency/model inventories are retained
before loading and checked again after requests; the GGUF identity identifies
the metadata input and does not imply that Pooled consumed its unused weights.

`referencesPath` contains `{ "prompts": [{ "id", "text", "promptIds",
"expectedTokenIds", "expectedText" }] }` for the exact four contract prompts.
`proofBinding` is `{ "path", "sha256" }` pointing to a pinned
`doppler.equal-weight-resident-numerical-binding.v1` packet. It contains
`dopplerSource`, `dopplerObservation`, `numerical`, and `prompts` (plus `pooledNumerical` for the subgroup numerical schema), where each
prompt ID maps to `doppler`, `pooled`, and `pooledSource` receipts; every reference
is `{ "path", "sha256" }`. Admission checks all native-reference and cross-engine
logit positions, exact tokens/text, inference configuration, complete original
runtime source (including its Git tree), native Pooled bytes and observer-only
deltas. It reuses numerical evidence, not historical resident materialization.
Missing or mismatched evidence blocks preparation.

Doppler delegates existing `initializeSuiteModel` and `runTextInference`;
its facade passes supported `options.inputIds` to native public `generate`.
Tokenization and reset/drain are outside timing. Native streaming text decoding
remains inside. Actual native token-delivery timestamps are captured immediately;
forwarding to the harness's fallback-text diagnostics is deferred until after
the final fence/end timestamp. Pooled uses native `prefillTokens` and
`forwardTokenIds`, then its native tokenizer decodes the buffered response once.
Both intervals return decoded text plus IDs and include a final queue fence.
Their native streaming/buffered API costs and delivery cadence remain explicit.
Output verification and flushed per-attempt files precede the next request.
Failures, partial output, warmups and cleanup errors remain in the output;
only a complete valid 20-pair cohort can produce a performance verdict.
