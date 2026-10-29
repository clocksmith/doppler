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

Focused checks: `node --test tools/pooled-equal-weights/canonical.test.js`.
Physical receipts and unresolved gates:
`reports/pooled-equal-weights-20261029/README.md`.
