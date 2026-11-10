# Numerical discontinuities: a diagnostic, not a replacement gate

The owner authorized a diagnostic experiment while preserving qualification.
This packet freezes two historical failure configurations and evaluates their
retained physical captures. **Neither model qualifies. No correctness contract
is proposed for promotion by this result.**

## New findings

### Flash-Next: a constructive rounding-order witness, with an insufficient bound

The original 2,304-position capture, artifact manifest, CPU-prefix fixtures,
loaded router weights, and raw capture bytes were verified by SHA-256 before
analysis. This is the historical F16-KV routing investigation, **not** a rerun of
the newer F32-KV candidate with 13 failing final logits.

We evaluated 12,288 GPU router logits and 12,288 CPU router logits: all 512
outputs at eight fixed positions in layers 0, 4 and 5. Exact dot products and
absolute-product sums use integers representing dyadic rational values; no
observed error is used to fit the bound. The conditional arithmetic envelope is
`gamma(2*K) * sum(abs(x*w))`, with an additional conservative underflow/flush
allowance. It assumes round-to-nearest operations and excludes overflow and
subnormal inputs. It is not a certificate of the GPU compiler's arithmetic.

All sampled logits lie inside their respective local envelopes. That is weak
evidence: the envelopes range from approximately 0.00197 to 0.00584, while the
largest sampled GPU dot-product error is approximately 0.0000289.

There is a stronger, constructive observation at layer 4, position 1017:

| Evaluation on the **same GPU input and weights** | Expert at the raw-logit top-10 boundary |
| --- | --- |
| Recorded GPU logits | 260 |
| Separate F32 multiply/add, forward accumulation | 260 |
| Separate F32 multiply/add, reverse accumulation | 218 |
| Exact dot products | 218 |

The exact 218-minus-260 gap is about 0.000009617. Changing only summation order
reproduces both orderings. This does **not** prove every GPU output correct:
forward replay matches 6,333 of 12,288 GPU logits bit-for-bit. These are raw-logit
rankings, not a new physical softmax/top-k replay, and the CPU and GPU upstream
inputs differ. The independently bounded upstream error remains unknown.

### KV: the store is correct on its actual inputs; earlier arithmetic remains open

The plain capture and paired reference proxy were verified against the retained
raw index. The corrected paired probes supply all 184 positions. The paired
intermediates are a high-precision proxy; their final cache identities were
previously matched to the approved F64 evaluator, but the intermediates are not
independent exact-real values.

An independent nearest-even F16 converter checked every sampled store:

| Layer / cache | Values checked per side | Incorrect stores, plain / proxy | Different stored values |
| --- | ---: | ---: | ---: |
| 3 / K | 94,208 | 0 / 0 | 343 |
| 3 / V | 94,208 | 0 / 0 | 311 |
| 7 / K | 94,208 | 0 / 0 | 3,775 |
| 7 / V | 94,208 | 0 / 0 | 3,995 |

The embedding captures agree across all 188,416 values. The first captured
arithmetic difference is layer-0 input normalization, maximum about 5.033e-7.
The changed cache values follow from different pre-store inputs, not an incorrect
cast. Observed input differences and crossed midpoints are measurements, never
substitutes for an independently derived upstream error allowance.

## Fault controls and validation

- 24 deliberately out-of-envelope router outputs were rejected.
- **24 deliberately altered outputs inside the envelope were admitted locally.**
  Envelope membership alone therefore cannot be the proposed acceptance rule.
- Four one-bit F16-store corruptions were rejected.
- Four corrupted pre-store inputs, correctly rounded afterward, remained
  unresolved at the upstream boundary. A correct store cannot certify its input.
- A changed input digest was rejected before analysis and produced no output.
- Six focused tests pass, including all 63,488 finite F16 encodings, all 63,486
  signed adjacent-value midpoints, and 128 fixed synthetic dot-product cases.

Fifteen router rows are reserved validation slices within the same already
studied prompt. They are **not unseen prompts or a blinded holdout**. No new
model-quality, long-decode, cross-device or performance campaign ran.

## What this resolves and what remains

The experiment demonstrates same-input arithmetic-order sensitivity and exact
cache casting. It also demonstrates why a generous local bound can forgive an
incorrectly altered output. It does not independently bound the complete
upstream recurrence, normalization, attention and residual chain in either
model. Flash-Next's earliest *routing-choice* divergence in the retained prefix
is layer 4; this is not the first arithmetic divergence in its entire graph.

The reference batch-geometry control uses the same captured inputs, F32 weights,
Torch version, thread count and precision setting. Its full-geometry replay
matches the frozen fixture exactly; only projection batch shape changes in the
other arms. These are equivalent real-valued matrix equations, with different
floating-point evaluation paths. This packet audits that retained control; it
does not rerun Torch or assert that bit identity was the intended contract.

A replacement contract still needs independent upstream bounds, an explicit
admissible-path definition, verified selection semantics, held-out prompts and
fault controls stronger than simple interval membership. The original campaigns
remain failed. The five comparator/reference files in
[qualification-preservation.json](qualification-preservation.json) match HEAD.

## Reproduction and retained identities

[protocol.json](protocol.json) was frozen before collecting and analyzing the
raw slices. [results.json](results.json) records every sampled row, boundary
example and fault-control result. [receipt.json](receipt.json) binds the tools,
protocol, results, tests and source evidence.

The 21,559,377-byte raw bundle is retained operator-locally on the analysis Mac
and Jupiter at `.cache/numerical-discontinuity-20261109/inputs.json.gz` under the
Doppler checkout. It contains weight slices and capture metadata and is not
published in this packet. Its SHA-256 is
`f3dad7bae18e1e801398892d85adbc96f3976a799cf670a6eee0f06045f881c3`.

From the Doppler repository:

```sh
node --test tools/numerical-discontinuity-core.test.js
node tools/analyze-numerical-discontinuity.js \
  reports/numerical-discontinuity-20261109/protocol.json \
  .cache/numerical-discontinuity-20261109/inputs.json.gz \
  f3dad7bae18e1e801398892d85adbc96f3976a799cf670a6eee0f06045f881c3 \
  /tmp/discontinuity-results.json
```

To reconstruct the bundle, `tools/collect-numerical-discontinuity.js` takes the
protocol, historical artifact directory, qualification directory, historical
split-capture directory, retained raw index, router-weight audit, and output
path. It requires Node and `unzip`, verifies the pinned historical bytes, reads
captured files and exits. It never loads a model or opens a GPU.

Component: `doppler.repository-tooling` and numerical diagnostic evidence.
Intent: preserved. Acceptance: focused arithmetic tests, hash-bound retained
capture analysis and negative controls. Boundary effects: reads historical
Reploid evidence; no runtime, reference, tolerance, service or deployment change.
