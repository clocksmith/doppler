# Remove repeated immutable Capsule hashing

The installed public generation profile localized substantial CPU time to
`getCapsuleIdentity` from the per-event execution guard. Loading already creates
a private, deeply frozen Capsule snapshot and verifies its signature, artifacts
and TargetPlan. Rehashing that same snapshot cannot detect a later change.

The candidate removes those repeated structural hashes. It preserves live device
availability, release authorization and program-identity checks at the same
execution boundaries. No sampling, shader, model, transport or trust policy
changes. `package-delta.json` proves that only
`src/client/runtime/composition-root.js` differs across the 1,796 packaged files.

## Installed comparison

- Baseline archive: `852b858a1d1f5d42a951f1bed747b8e45d63c3bfad95f4e8c2e186a6d1affbd2`.
- Candidate archive: `f9758530d0fec3967dfa96dc64252dcdd874c0c73f120ee28f7767899f0ca2d6`.
- Same Qwen 4B Capsule, TargetPlan, 319-token prompt, request settings, physical
  AMD Radeon 8060S and Chrome 146.0.7680.177.
- Baseline, candidate, then baseline again; each loaded session executes a first
  request, an instrumented repeated request, and an uninstrumented repeated
  request. All nine outputs match exactly: 107 tokens and EOS completion.
- Uninstrumented decode medians: baseline **89.1–90.7 ms**, candidate
  **73.55–74.85 ms**. The baseline returns to its original range afterward.
- Instrumented SHA-256 inner-function CPU samples fall from about 2,514 ms to
  663 ms. Remaining live execution-identity hashing is preserved.
- First-token latency does not improve: approximately 12.74–13.04 seconds for
  baseline and 13.13–13.22 seconds for candidate. The reduction is in decode
  CPU overhead, not GPU arithmetic, loading or prefill.

`comparison.json` retains all measurements. This is a bounded local mechanism
comparison, not a randomized vendor benchmark or a universal speedup claim.
The 128-token request stopped at EOS after 107 tokens; it does not satisfy a
128-token shader qualification gate. No shader changed here.

## Correctness and acceptance

The regression proves nested metadata cannot be mutated, changing caller-owned
input cannot change the loaded snapshot, and changing live program identity
while a stream is paused rejects before the next decode phase. Existing dtype,
fusion, memory-policy, graph, cancellation, loss and lifecycle tests remain.

All 20 runtime test files pass. Installed package exports/types, injected
standalone operations, and Reploid v1/v2 generation, streaming, adapters,
cancellation and ownership pass. Public boundaries, source architecture/style,
declarations and current runtime closure pass. Physical standalone results and
raw profiles are retained here and in `../capsule-profile/`.

The registry routing audit was also regenerated after the interface repair:
30 manifests and 136 opportunities remain, and all 76 previously surfaced
integrity failures are unchanged. Updated descriptor digests identify current
registry metadata; historical model manifests and qualification receipts were
not rewritten.

## Reproduce

With the exact installed bundles and provisioned model paths in the configs:

```sh
node tools/profile-installed-capsule.js artifacts/runtime-coherence/capsule-profile/config.json
node tools/profile-installed-capsule.js artifacts/runtime-coherence/capsule-profile/candidate-config.json
node tools/profile-installed-capsule.js artifacts/runtime-coherence/capsule-profile/baseline-after-config.json
node artifacts/runtime-coherence/immutable-execution/compare.js
node tools/run-node-tests.js tests/runtime
```

Use a writable `TMPDIR` and fresh output directories. The configs retain the
same requests, local model locations and trust keys. Candidate and bracketing
baseline receipts identify their archives; archive versions alone are insufficient.
The copied `baseline-after.cpuprofile` corresponds to that receipt's original
`run-1.cpuprofile` name.

Component: `doppler.runtime-source.client`, `doppler.tests`, `doppler.docs`;
current routing inventory: `doppler.benchmarks`.
Intent: preserved.
Acceptance evidence: `comparison.json`, `package-delta.json`, package and
installed-consumer receipts, runtime tests and checks retained here.
Boundary effects: validation lifetime of immutable metadata only. Publication,
deployment and signed model contents are unchanged.
