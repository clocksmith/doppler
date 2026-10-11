# Typed resident output-allocation failure

Component: `doppler.runtime-source.inference.pipelines.text`. Intent: preserved.
Boundary effects: one public error code; generation, stopping, budgets and cleanup
are unchanged. `DOPPLER_RESIDENT_OUTPUT_LIMIT` identifies character-allocation
failure. It is an error, never EOS or successful token-budget completion.

The physical same-device resident test executes real model layers, triggers a
one-character allocation denial, checks the typed error and settles both attempts.
Its six ordinary split/unsplit steps and existing lifecycle controls also pass.
`physical.json` binds the model, generation, selected provider and edited source.
This is a Node WebGPU resident diagnostic, not two-device or signed-Capsule proof.

Validation: the focused resident lifecycle/opening tests passed. `npm run
check:green` completed 895 test files and reported three packaging failures:
stale generated closure, the same stale receipt in the installed smoke, and
186 added bytes above the recorded source-size inventory. The closure was
regenerated; the exact 1,897-file package inventory was retained and the payload
accounting updated by those 186 bytes. The three failed checks then passed via
`node tools/run-node-tests.js --scripts runtime:closure:check public:boundaries:check package:smoke`.
Numerical thresholds and physical resource limits were not changed.

The existing 0.6.27 evaluation archive was not rebuilt or replaced. Reploid's local
adapter recognizes its exact older untyped error through a fixed compatibility
allowlist; new source callers receive the typed code. A future archive still needs
the applicable installed and physical qualification before promotion.
