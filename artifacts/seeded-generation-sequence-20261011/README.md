# Seeded generation advances by output position

The actual Capsule session controller repeated the same random draw at every
step. With four equal logits and seed 17 it emitted twelve zeros. The resident
and direct-pipeline owners also forwarded an unchanged seed. The repair derives
each effective seed from the immutable request seed and generated-token index,
using the existing scalar sampler. The same Capsule reproduction now emits
`[0,0,3,1,2,1,3,0,1,2,3,0]`.

`tests/inference/seeded-generation-sequence.test.js` exercises real Capsule and
resident orchestration with synthetic logits. It covers independent interleaved
attempts, same-seed reproducibility, repeated-index retries, fresh attempts after
cancellation, unchanged admission digests, forced-token advancement, CPU and
declared GPU selection boundaries, direct-pipeline owners and greedy controls.
GPU execution is stubbed in this test; it is not physical inference evidence.
The existing scalar sampler and GPU shader arithmetic remain unchanged.

Output index zero preserves the old first draw. Later steps use
`((seed % 4294967296) + (index % 4294967296)) % 4294967296` so large finite seeds
cannot absorb small increments. Tests cover zero, fractional seeds, the integer
precision boundary, `Number.MAX_VALUE`, wrap, rejected negative/null seeds and
unseeded operation. This does not claim sequence equivalence with PyTorch.

The initial package smoke correctly rejected the stale source closure receipt.
The existing closure generator refreshed it, and the retained rerun passes the
installed package's public imports, declarations, resident contracts and
synthetic signed-Capsule execution checks. The temporary installed snapshot is
SHA256 `4e1f62a742cc7b1dc8ac0573964e701c75463069f124f6bc52f9d7e258c2dac0`.
Its source-state and receipt are retained here. This test-only snapshot also
contains the already-committed numerical source repair; neither this check nor
its version number qualifies those physical changes.
The exact sequence regression also passes against the installed files with its
imports relocated mechanically into the consumer; it uses no source-checkout
runtime. That internal behavior check is separate from the package smoke's
public-entrypoint and declaration checks.

The accepted standard archive remains unchanged. Seeded multi-token behavior
requires requalification against the new source; historical results are retained.
Type, style, architecture and the focused sampler/lifecycle controls pass.
The full unit suite passes all 901 files; its result is retained in `unit.log.gz`.

Component: `doppler.runtime-source.inference`. Intent: preserved. Boundary
effects: seeded output sequencing corrected; immutable request identities,
greedy output, model arithmetic and checkpoint capabilities unchanged.
