# Installed Capsule generation profile

Three requests through the unchanged installed archive
`852b858a1d1f5d42a951f1bed747b8e45d63c3bfad95f4e8c2e186a6d1affbd2`
produce identical complete output: 107 tokens, stopped by EOS with a 128-token
limit. The first 16 match the pre-existing frozen reference. The remaining tokens
establish repeatability, not new source-model qualification or a 128-token gate.
The model, signed TargetPlan, prompt, options, physical AMD adapter and Chrome
version are retained in `receipt.json` and `config.json`.

This is a diagnostic population: first request, instrumented repeated request,
and uninstrumented repeated request in one loaded session. It is not an
interleaved baseline/candidate throughput comparison. The public stream
accumulator verifies every completion and display text.

Observed on the instrumented request:

- One 607,744-byte score readback per token; 64,420,864 bytes across the 106
  decode steps, plus the first token's scores and one 16-byte finiteness check.
- Decode uses 436 dispatches and six submissions per token.
- CPU stack sampling estimates 93.8 ms inside `sampleCapsuleLogits` across the
  whole request. Penalties were explicitly disabled by this frozen workload.
- SHA-256's inner function accounts for about 2,514 ms of sampled CPU time.
  Its dominant callers include `getCapsuleIdentity -> assertPlanUnchanged`,
  repeatedly validating and hashing the session-owned frozen Capsule.
- First token: about 12.75 seconds. Decode median: about 90.9 ms under
  instrumentation, 89.1 ms in the subsequent uninstrumented request.

These observations support investigating redundant immutable metadata hashing
before attributing most CPU overhead to sampling. GPU readback remains a
separate measurable target. Queue/map waits overlap; `analyze.js` reports their
union and preserves individual observations without claiming GPU kernel time.
CPU samples are estimates, include observation overhead, and are not additive
with the wait ledger. No performance improvement is claimed by this baseline.

Reproduce with the provisioned exact model and installed bundle:

```sh
TMPDIR=/dev/shm/doppler-runtime-coherence node tools/profile-installed-capsule.js artifacts/runtime-coherence/capsule-profile/config.json
node artifacts/runtime-coherence/capsule-profile/analyze.js artifacts/runtime-coherence/capsule-profile/receipt.json
node tests/tooling/installed-gpu-observation.test.js
```

Use a new `outputDir` when rerunning; the tool does not overwrite runs.
`probe-executed.js` and `observer-executed.js` retain exact executed source bytes
named by the receipt hashes. They are source snapshots, not standalone entry
points. The maintained probe additionally closes the session on failure.

Component: `doppler.repository-tooling`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: physical `receipt.json`, raw `run-1.cpuprofile`,
`summary.json`, and observation behavior test.
Boundary effects: optional development observation only; unchanged runtime,
model artifacts, public contracts and deployment.
