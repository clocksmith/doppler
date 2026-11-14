# CATSCAN: Observatory

Component: `doppler.observatory`

Parent: [Doppler Repository](../CATSCAN.md)

## Target

Let people see how an answer forms: ask, watch it stream, inspect any token inside the model, branch, compare runs, and benchmark a device, in one coherent browser site.

## Authority

- Owns the Explore and Benchmark pages, the saved-run format, its views and
  signal math, and benchmark-result comparison.
- Does not own runtime algorithms, inspection capture, model facts or the
  benchmark's acceptance thresholds beyond its versioned protocol file.

## Scope

- `public/`, `config/`, `server/`, `scripts/` and `test/` in this folder.

## Contracts

- Input: the public runtime API (`dr.load`, `inspect.generate`,
  `inspect.projectLayerOutputs`) served from `/doppler/src/`, and the demo's
  styles and recorded example under `/doppler/demo/`.
- Output: user-visible runs, `doppler.observatory.run/v1` files and
  device-benchmark result files.

## Invariants

- Every signal and view shows only what a run recorded; unrecorded values are
  marked, never estimated beyond their label.
- Answer comparison and benchmark-result comparison stay separate operations.
- Instrumented runs are never presented as speed measurements.
- Saved runs from earlier versions keep opening; retired addresses redirect.
- Nothing is uploaded.

## Acceptance

- `npm test` and `npm run check:workspace` pass in this folder.
- Evidence: [Observatory tests](test) and [browser check](scripts/check-workspace.js).

## Freedom

Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.

## Non-goals

- A second runtime, rankings, or uploads without a backend, consent and abuse handling.
