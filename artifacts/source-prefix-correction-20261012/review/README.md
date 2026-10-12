# Independent product-residual review

Component: `doppler` (numerical investigation evidence). Intent: preserved.
Boundary effects: none; no runtime, reference contract, precision, or shader edits.

## Reproduce

From the Doppler checkout:

```sh
node artifacts/source-prefix-correction-20261012/review/product-residual-cpu-review.js
```

The checker rejects a changed candidate SHA before running. Its output binds the
checker SHA, candidate SHA, Node version, seed, PRNG, input count and lane counts.
The retained JSON records **931,639 comparisons and limb-product checks, zero
mismatches**, including eight explicit midpoint/sign cases and sixteen signed
regressions for subnormal inputs, flushed products and upward underflow rounding.
It samples 500,000 pairs of raw binary32 bits using xorshift32 seed 624901.

The candidate translation uses Number/U32 limbs and wrapping bit operations.
The independently structured reference uses signed BigInt dyadic coefficients,
exact multiplication/subtraction and integer division with nearest-even rounding.
Every limb multiplication is additionally checked against exact BigInt
multiplication. Binary64 multiplication selects the two enclosing binary32
products; a binary32-by-binary32 product fits exactly in binary64. The checker
tests nearest-even, the other finite adjacent result and synthetic flushed-zero
products when an operand or product is subnormal. Synthetic FTZ is not an
observation of GPU behavior. Exact-zero residual signs are equivalent; nonzero
underflow residual signs remain checked.

An earlier ephemeral Python diagnostic reported 931,392 passing comparisons,
seed 624901 and 500,000 operand pairs. Its command was not retained as a file.
That count is historical chat-only evidence, **not reproduced by this script**.
The retained JavaScript run uses a declared different PRNG and a stronger exact
integer reference. Do not cite the old count as a reproducible acceptance result.

## Review findings

The review identified two gaps in the preceding candidate/harness: supplied CPU
products did not exercise GPU product flushing, and nearest-even-only inputs
missed valid adjacent rounding near underflow. The range candidate addresses
both. The GPU harness owner independently added actual-product and adjacent
rounding lanes; their physical receipts remain outside this CPU receipt.

The 16/8-bit limb split keeps each partial product and cross-term within U32,
and one low-word carry reconstructs the exact 24-by-24-bit product. Two-word
subtraction retains the residual needed for subnormal and FTZ controls.
The rounding helper handles low-word sticky bits and nearest-even ties; carry
from subnormal to minimum normal and normal significand renormalization are
consistent with the exact reference. The `shift >= 64` product branch returns
the negated supplied product because the exact term is below residual rounding
resolution in this reachable domain.

This is a bounded-domain review, not a formal exhaustive proof. Inputs with
nonfinite operands or nearest-even product overflow are excluded and counted.
Arbitrary unrelated supplied products are not accepted inputs to the helper.
`rounded_shift` is not a generic U64 utility: returning zero at shift 64 relies
on the reachable magnitude bound; arbitrary U64 values above 2^63 need different
rounding. `scaled_significand` likewise has no general overflow contract.

The [WGSL floating-point contract](https://www.w3.org/TR/WGSL/#floating-point-evaluation)
permits implementation rounding choices and subnormal flushing. CPU translation
does not establish that a compiler preserves the shader's behavior. Physical
GPU receipts remain required, followed by unchanged full-model acceptance.
No performance, whole-model qualification, runtime promotion or deployment is
established by this review.
