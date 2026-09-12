# Release kernel-digest audit

The npm `doppler-gpu@0.6.1` archive differs from the retained September 8
candidate in 34 conversion/source-package JSON files. Runtime JavaScript and
WGSL match. The published archive still has stale `silu.wgsl#main` and
`sample.wgsl#sample_single_pass` digests in the Qwen 3.5 9B f16 recipe.
The [archive comparison](archive-comparison.json) and
[published-package rejection](published-rejection.log) retain this finding.

## Fix

Branch `fix/release-kernel-digest-validation` synchronizes 72 stale digests in
35 checked-in recipe files. No runtime JavaScript, WGSL or retained model
manifest changes. The [exact digest rebindings](recipe-digest-rebindings.json)
also constrain historical regression assertions; prior recipe/manifest and
Capsule migration evidence remain intact. The updated semantic-lowering
receipt is versioned under `reports/release-kernel-digests/`.

`npm run kernels:package-digests:check` now runs in the default green chain.
`package:smoke` checks the installed archive's recipes against its digest
registry. Checks do not rewrite installed packages. Source-only synchronization
does not touch `models/local`. Missing package directories, malformed JSON and
an empty digest registry fail. Pending, unregistered kernels are outside this
registered-digest consistency check and receive no support qualification.

The measured package remains 1,784 files. Adding the two script references
increases `package.json` by 170 bytes; the unpacked-size ceiling is updated to
the measured 10,883,619 bytes. The packed-size ceiling is unchanged. See the
[package audit](package-audit.json).

## Acceptance

The [machine-readable audit](audit.json) identifies the clean source commit and
hashes the [complete green-chain output](check-green.log).

- `npm run check:green`: passed, including all 798 unit-test files and package
  import, CLI, installed-recipe, Electron, embedding and TypeScript checks.
- [Focused digest/package regressions](focused-tests.log): three files passed.
- [Updated historical/semantic regressions](fixture-tests.log): four files passed.
- Reploid `6478fc3e9b54447ebf0f7e53a094369ee7c4cecf` pins the actual published
  `0.6.1` package and browser URLs. It passes 2,506 unit tests, four document
  browser tests and one real local ESM-2 Chrome/WebGPU execution and reload test.
  Its release record is
  `artifacts/doppler-release-061-2026-09-08/2026-09-12/README.md` in that repository.

Initial failed runs remain at
`/var/tmp/doppler-061-registry-20260912/check-green.log` and
`check-green-final.log`: the first exposed the exact package-size increase;
the second exposed four stale fixture expectations and eight failures after
the `/tmp` user quota was exhausted. The final invocation uses an isolated
`/var/tmp` temporary directory. Old evidence was preserved, fixture expectations
bind only the explicit digest corrections, and generated lowering evidence has
a new path. No test was skipped to obtain the passing result.

These Doppler fixes are unpublished source changes; the registry package still
contains the two defective recipe pins. Reploid's physical smoke uses the
published runtime with its existing signed ESM-2 Capsule and does not convert
the affected Qwen model. No npm publication or production deployment was
performed during this follow-up.

Component: `doppler.runtime-source.config`, `doppler.repository-tooling`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: the commands, logs and audit above.
Boundary effects: release validation and generated recipe identities; paired
Reploid package/browser/model version pins. Runtime execution authority is unchanged.

*Last updated: September 2026*
