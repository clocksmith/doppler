# Selective materialization diagnosis

Component: `doppler.repository-tooling`. Intent: preserved. Boundary effects:
diagnostic tooling and retained evidence only; runtime, package pin, approved
reference, weights, F16 cache storage and 0.001 tolerance are unchanged.

The audit now selects F32 stores by operation, layer and prefill/decode phase,
and separately controls scalar parameter rounding. It preserves full causal
token history, records changed cache bits and rejects unused selectors. The
approved evaluator is never edited. Every run first reproduces all 51 approved
request-2 vectors exactly with the inactive diagnostic copy.

## Results

CPU runs on x@128 used four BLAS threads and low scheduling priority. No GPU
participant was stopped. The eight-prefix baseline reproduces the previous
mapped vectors, store counts and cache identities exactly. All later default
51-prefix baselines also reproduce each other exactly.

| Control | Prefixes failing | Maximum absolute logit error |
|---|---:|---:|
| Original mapped stores | 2/8 | 0.001502511656 |
| Remove stores through layer 6 | 0/8 | 0.000472576370 |
| Remove normalization and projection stores | 0/8 | 0.000894154953 |
| Mapped stores, longer history | 24/51 | 0.003091770105 |
| Remove normalization and projection stores, longer history | 1/51 | 0.001454645714 |
| Remove all non-state F32 stores, longer history | 1/51 | 0.001394484393 |
| Same, but retain source recurrent RMS epsilon | 0/51 | 5.39239764e-11 |
| Retain source recurrent RMS epsilon with ordinary mapped stores | 22/51 | 0.003513270518 |

The final row remains a failure, not a proposed epsilon-only repair. The
51-prefix runs extrapolate the reviewed operation map beyond its eight captured
prefixes; they are CPU counterfactuals, not physical execution qualification.

The scalar control identifies a concrete conditional cause. With non-state
stores suppressed, rounding recurrent readout RMS epsilon changes layer-3 key
cache coordinates `[84, 1, 165]` and `[233, 0, 51]` (token, head, dimension):
expected/actual F16 bits are `35159/35160` and `47964/47963`. Preserving source
epsilon removes these differences and reproduces all approved cache identities.
Preserving only ordinary RMS epsilon or recurrent L2 epsilon does not remove
the failure. Ordinary intermediate stores still cause failures after this
scalar repair. Several smaller changes improve short prefixes but worsen later
ones; all outcomes are retained.

This identifies interacting storage and scalar boundaries. It does not establish
a unique production cause, a minimal correction, an attainable GPU oracle, or
permission to increase precision. The [inactive proposal](inactive-precision-proposal.json)
specifies the additional implementation authority needed before broader runtime
precision work. Signed physical opening and release qualification remain open.

## Evidence and reproduction

The `*-cases.json` files specify each intervention; `cases.json` is the initial
batch. Matching `*-summary.json` files preserve metrics, selectors, selected
element counts, cache differences and individual vector hashes. Full vectors
and original logs remain on x@128 at the exact paths and hashes in
[raw-index.json](raw-index.json). Earlier executed tool versions are retained as
compressed source beside this receipt; later runs use the committed tool.

From the Doppler checkout on x@128, use the existing reference dependency path:

```sh
PYTHONPATH=dist/decision-acceptance/reference-dependencies \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 nice -n 10 \
python3 tools/qwen35-short-prefix-audit.py \
  --capture reports/local/two-f32-candidate-20261012/q4-decode7-capture.json \
  --contract artifacts/whole-model-f64-complete-20261011/request-2-contract.json \
  --binding artifacts/recurrent-two-f32-20261012/q4-model-binding.json \
  --ideal-materialization-map artifacts/recurrent-two-f32-20261012/ideal-materialization-map.json \
  --materialization-interventions artifacts/selective-materialization-20261012/cache-coordinate-cases.json \
  --materialization-prefix-count 51 \
  --out reports/local/selective-materialization-20261012/reproduction.json
```

Omit the prefix-count option for the original eight-prefix scope. Preserve
failed experiments; none of these results is a replacement acceptance reference.

Validation: `python3 tests/integration/qwen35-short-prefix-audit-test.py`, Python
compilation and `git diff --check` pass. The first remote launch failed before
computation because its Python path omitted `gguf`; `missing-gguf.log` preserves
that failure. No new dependency installation was needed.

## Inactive proposal memory review

The [source-bound memory inventory and review](memory-review.md) calculates a separate-low-buffer design for two 1,588-token attempts. The [result](memory-calculation.json) separates active and reusable storage, the already approved recurrent-state cost, unchanged F16 KV, and host activation payloads. It is not a measured concurrent peak or permission to implement broader precision. Regenerate and verify with `node artifacts/selective-materialization-20261012/calculate-memory.js --check`.
