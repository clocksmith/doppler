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
   snapshot. It captures full first-step logits and optional single-token
   boundaries after the benchmark request. Timings from this snapshot are
   inadmissible. Local model caches are excluded and dependencies are linked.
5. Compare prompt IDs, generated IDs, full numerical outputs, backend identities,
   and effective weight materializations before starting the timing contract.

The shared contract is
`benchmarks/vendors/pooled-equal-weights/contract.json`. Machine paths, browser
arguments, deadlines, and output directories are explicit run JSON fields;
retained examples are in the report archive. Never reuse an existing output.
The supplied vendor tree is read only, and its actual bytes—not an enclosing
repository's Git revision—are the source identity.

`rne-probe-browser.js` and `kv-cache-rne-prototype.wgsl` are an independent
rounding experiment. The prototype is not registered or shipped as a runtime
kernel. Both native and prototype conversion passed the retained Python
binary16 oracle; this experiment did **not** establish the cause of the model
mismatch. Optional prototype replacement is confined to an explicitly recorded
diagnostic snapshot and cannot supply performance evidence.

Focused checks: `node --test tools/pooled-equal-weights/canonical.test.js`.
Physical receipts and unresolved gates:
`reports/pooled-equal-weights-20261029/README.md`.
