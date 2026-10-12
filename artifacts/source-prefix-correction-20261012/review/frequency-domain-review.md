# Independent frequency-residual and extent review

Component: Doppler numerical evidence. Intent: preserved. Boundary effects:
none; this review does not modify shaders, runtime, reference authority or GPUs.

Run `node artifacts/source-prefix-correction-20261012/review/frequency-domain-review.js`.
The retained JSON binds the exact experimental shader, checker and Node version.
The frequency candidate remains rejected by its separately retained whole-model
results. This review cannot override those results.

## Arithmetic domain

The logarithm uses the atanh expansion through degree 27 after mantissa
reduction. On the nominal exact reduction interval, its tail is bounded by
`2*s^29 / (29*(1-s^2))`, approximately 4.47e-24. The exponential Taylor polynomial
through degree 14 has a remainder below approximately 1.35e-19 on
`[-ln(2)/2, ln(2)/2]`. These analytic truncation bounds exclude coefficient,
range-reduction, operation and backend rounding errors. They are not claimed
GPU accuracy bounds. The compensated pair operations still use F32 operands
and temporaries; table bindings and outputs remain F32. No F64 storage is added.

The first-order phase correction is **not validly bounded below F32 rounding
across its enabled position range**. A concrete permitted input is theta 2,
base dimension 256, dimension 125, position 16382. Its high angle is
8325.154296875 and its residual approximately 0.0004805606168. Even using ideal
host-double sine and cosine of the high angle, the first-order cosine differs
from cosine of the full angle by approximately 1.152e-7 and rounds to a
different F32 value. The script reproduces this witness independently of GPU
behavior. It isolates the missing quadratic term, not an error in log/exp.

The numerical owner received this finding. A future candidate would need a
qualified higher-order correction or a justified narrower operation domain;
this review does not change the recorded candidate. Arbitrarily short positions
also do not guarantee identical nearest rounding near a rounding boundary.

The explicit candidate guard excludes scaled and externally supplied-frequency
paths and restricts theta to positive normal values from 1 through about 1e30.
Very small low terms can nevertheless become F32 subnormals; the analytic
polynomial bounds do not prove compensation survives backend flushing.
CPU binary64 `pow`/trig supplies a robust counterexample, not a replacement
independent whole-model evaluator. Kernel lint emitted only its generic builtin
`vec3` warning; generator syntax passed.

## Standalone table extent

The concurrently reviewed extent repair correctly separates `positionCount`
from declared `maxSeqLen`. Allocation, dispatch, bounds and cache identity use
the former; LongRoPE factor selection and amplitude scaling retain the latter.
The reviewed controls verify model context 8192, allocated positions 8, long
factors selected despite short allocation, and separate cache identities when
either context or extent changes. Standalone and partition initialization are
both covered. Invalid extents are rejected.

MRoPE auxiliary positions retain their existing full-context axes and stride,
so the observed change reduces cosine/sine table storage, not all auxiliary
storage. Both CPU controls were independently rerun successfully:

```sh
node tests/inference/rope-frequency-cache-contract.test.js
node tests/gpu/rope-precompute-dispatch-contract.test.js
```

These use fake devices and inspect uniform/allocation contracts. They do not
qualify physical LongRoPE output or model answers. The numerical owner retains
the actual source changes and physical receipts separately.
