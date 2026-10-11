# Independent cache-producing midpoint audit

The focused offline audit covers the first full-attention layer, layer 3, across
all 187 positions through request 2, decode step 3. It uses the existing physical
capture; no additional GPU run, production arithmetic, precision, reference, or
0.001 threshold changed. All eight source-model prefixes remain bit-identical to
the prior untouched source computation. Its selected logit error remains
**0.0014121532440185547**, so numerical qualification remains open.

The source observer retains independently evolved operands without replacing
them. Separate Float64 equations evaluate K/V projections, key normalization,
and rotate-half text RoPE on each implementation's own incoming values. These
actual-operand checks distinguish local evaluation differences from differences
already present at an operation's inputs. The captured GPU values never enter
the independent model's state.

| Boundary | Values | Different F16 results | Difference persists with own-input Float64 math | Both match their own Float64 rounding |
| --- | ---: | ---: | ---: | ---: |
| Value projection | 95,744 | 661 | 512 | 473 |
| Key after RoPE | 95,744 | 685 | 650 | 646 |

These are the actual cache storage boundaries. Additional key projection and
normalization checks are retained as intermediate diagnostics; those outputs
are not themselves stored as F16. Counter columns describe overlapping subsets,
not additive error contributions.

Maximum local GPU/source errors against each side's own Float64 calculation are
1.916e-6/1.582e-6 for value projection, 1.045e-6/1.180e-6 for key normalization,
and 1.594e-5/1.785e-5 for key RoPE. RoPE uses independently calculated Float64
frequencies, so its local difference includes F32 frequency/trigonometric
evaluation. These figures do not identify a one-sided implementation defect.

For example, value coordinate `[122,179]` has a midpoint of `1.75341796875`.
Independent own-input calculations produce `1.7534173527967` on the GPU side
and `1.7534183660496017` on the source side: their incoming operands already
put them on opposite sides. Their actual F32 values round to the respective
Float64 result's F16 value. Elsewhere, small local rounding errors cross a
midpoint on either implementation. A lower isolated operation error does not
guarantee matching the other implementation's subsequent stored cache.

The [review](review.json) binds deployed weights, tokenizer, source configuration,
the existing GPU capture, eight unchanged source logits, and the full local
report. It retains bounded samples with position, head coordinate, actual and
Float64 values, rounded values, and exact adjacent-half midpoints. Together with
the separate physical-cache check, this closes the selected storage hypothesis
and the first-layer cache-producer attribution gap. It does not qualify every
prefix or establish an approved numerical reference.

The defensible next disposition is to define and approve the reference's
evaluation/rounding contract explicitly. Correcting the historical undeclared
normalization dispatch can establish a new implementation regression baseline;
captured output alone cannot become independent accuracy evidence. Preserve the
historical failures, independent-source failure, and 0.001 requirement.

Reproduction from the repository root:

```bash
PYTHONPATH=/tmp/doppler-reference-deps python3 tools/qwen35-deployed-reference.py \
  --model models/local/qwen-3-5-0-8b-q4k-ehaf16 \
  --source-config artifacts/source-model-reference-20261010/source-config.json \
  --capture reports/local/frozen-reference-producer/doppler-current026-full-linux.json.gz \
  --reference ../reploid/tests/fixtures/distributed-reference.json.gz \
  --piece-index ../reploid/self/config/model-pieces/qwen-0-8b-pieces.json \
  --piece-index-identity sha256:18adb1f08f4e694d357a27f7a5d67ea57a7d50c5e82aa7fda099084b445974b7 \
  --prefixes 8 --threads 8 \
  --boundary-capture reports/local/source-model-reference-20261011/all-attention-gpu-binary.json \
  --cache-midpoint-audit --out reports/local/cache-midpoint-20261011/comparison.json
```

Synthetic controls cover inherited versus local midpoint crossings, invalid
shapes/nonfinite values, JSON serialization, closed-form rotation, untouched
nonrotary coordinates, observation immutability, and hook removal on failure.
The original report-serialization failure and its retained prefix file binding
remain in [failed-run.json](failed-run.json); the corrected invocation completed.

Component: `doppler.repository-tooling`, `doppler.tests`. Intent: preserved.
Acceptance: `PYTHONPATH=/tmp/doppler-reference-deps python3
tests/integration/qwen35-deployed-reference-test.py`, Python compilation,
eight-prefix output identity, and diff checks. Boundary effects: offline
observation and analysis only; no runtime or reference activation.
