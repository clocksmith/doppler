# Independent deferred-half midpoint review

Component: `doppler` numerical investigation evidence. Intent: preserved.
Boundary effects: none; no runtime, shader, reference-authority, or GPU changes.

Reproduce from the repository root:

```sh
node artifacts/source-prefix-correction-20261012/review/half-midpoint-cpu-review.js
```

The script checks the exact candidate shader SHA before running. Its retained
JSON binds that SHA, checker SHA, Node version, input enumeration and results.
It reports **571,392 comparisons, zero mismatches**. There is no random seed:
the domain enumerates all 31,744 positive binary16 adjacent-value midpoints,
including the finite/overflow threshold, both signs, the midpoint and its two
adjacent binary32 values, and residuals of minus/zero/plus one quarter binary32
ulp. This covers zero/minimum subnormal, largest subnormal/minimum normal,
normal exponent transitions, odd/even ties, and the ±65520 overflow midpoint.

The helper is a literal bitwise translation. The independent reference searches
an ordered table of binary16 values, compares distances and resolves ties to the
even encoded significand. The virtual successor 65536 establishes the overflow
threshold. Each tested high-plus-low fits exactly in binary64; the reference
therefore avoids the binary32 double rounding that the helper is intended to
repair. An initial ephemeral Python `struct` binary16 check tested the same
571,392 cases successfully; the retained JavaScript script is the reproducible
evidence and uses a separately structured reference.

## Findings

The helper preserves the correct side of a midpoint for the tested finite,
nonoverlapping high/low inputs. Sign-directed increment/decrement is correct
for negative values as well as positive values. Returning the unchanged high
when low is zero leaves final nearest-even rounding intact. The exponent range
102–142 includes the first underflow midpoint and the last overflow midpoint;
the shift changes correctly at minimum-normal binary16.

One call-site defect was identified in the executed recipe
`prepare-deferred-half-candidate.js`, bound to archive `3035f07b…`:
its activation checks `kvDtype === 'f16'` but does not require
`state.kvCache.hasGPUCache()`. The actual cache-write branch in
`attention/interpreter.js` requires that predicate. When it is false,
`resolveKVCacheState` uses current F32 K directly. A configured but unallocated
cache can therefore expose a nudged F32 value without the promised F16 store.
The numerical owner acknowledged adding the guard to a new candidate identity;
this review does not rewrite or qualify the historical recipe.

The inspected geometry guards otherwise exclude retained/shared KV, aliased V,
skipped writes and DiffusionGemma decoder paths. Equality of the pair span and
rotary dimension ensures all rotated coordinates lie below the declared start
for either interleaved or rotate-half RoPE. The wrapper rejects noninteger,
negative, out-of-head start values and non-F32 K. Fused cache-writing projection
paths already report QK normalization applied, excluding this standalone branch.
These are source-review findings, not executed geometry controls.

Kernel lint passed with a generic `vec3` warning: the uses are invocation and
workgroup builtins, not buffer fields. The candidate builder passed `node --check`.

## Limits

This does not establish that the normalization approximation supplies the true
residual sign relative to the approved F64 equations. That requires independent
operator and whole-model evidence. Arbitrary large, overlapping or nonfinite
low parts are outside this helper contract. CPU IEEE binary16 behavior does not
prove physical GPU subnormal/overflow behavior. No performance, package,
application or numerical acceptance claim follows from this review alone.
