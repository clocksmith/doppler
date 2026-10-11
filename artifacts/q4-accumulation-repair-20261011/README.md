# Bounded Q4 projection accumulation

The executed F32-output wide-tile Q4 projection now reduces each 256-weight
quantization block before adding its subtotal to the output. This preserves
raw Q4 decoding, F32 arithmetic, layouts, dispatch geometry and selection.
Each row uses four local FMA lanes and one outer scalar, instead of four
uninterrupted lanes across the complete reduction.

This is a general local-accuracy repair, **not full numerical qualification**.
The standard 0.6.27 archive and deployed model manifest remain unchanged.
The independent F64 evaluation remains a diagnostic, not an activated reference.

## Evidence and limits

The fresh native-pipeline observer identifies `fused_matmul_q4_widetile.wgsl`,
entry `main`, default TILE_M=4/TILE_N=256, SHA-256
`a38c97f826a93929e40ce246ec5abf465bbd5d96863a2556d261dbaa6b325555`.
The first QKV/Z dispatches are [24,46,1] and [8,46,1]. All four baseline logits
remain byte-identical with the observer enabled.

Interleaving low/high nibble groups in the CPU F32 FMA emulator exactly reproduces
49,152 captured QKV values and 16,384 Z values on actual normalized operands.
This establishes accumulation order and decoded-weight behavior for these
operands; an earlier generic strided emulator did not match that order.

The physical AMD/RDNA3 Chrome 146.0.7680.177 replay reproduces every baseline
value. Block subtotals reduce QKV maximum local error from 7.658228e-6 to
2.889857e-6, and Z from 2.113094e-6 to 1.027539e-6. Five additional shape
controls pass, covering partial rows/columns, a 257-element tail, K=4096,
negative alpha, and TILE_M=8/TILE_N=128. Existing baseline and candidate
outputs, unisolated timings, and independent F64 expected values are retained.
No performance claim follows from those timings; other GPU contexts existed.

Against independently evolved F64 computation with F16 cache boundaries:

| Prefix | Tiled F16 repair only | Plus Q4 repair | Values above 0.001 |
| --- | ---: | ---: | ---: |
| 0 | 0.0006071924 | 0.0007045465 | 0 |
| 1 | 0.0008898752 | 0.0009503841 | 0 |
| 2 | 0.0007923877 | 0.0008746289 | 0 |
| 3 | 0.0012063517 | 0.0012218348 | 11 → 2 |

The first three maximum and RMS errors increased; the fourth maximum also
increased slightly. Local accuracy does not guarantee monotonic whole-model
agreement, particularly across discrete F16 cache rounding boundaries. This
result does not prove which boundary causes the residual disagreement, or
that the disagreement is unavoidable. Tokens remain 16,198,17,198.

The remaining failing vocabulary indexes are 18050 (+0.0012218348) and
202605 (-0.0010693408). Historical-reference failure remains retained.
Neither all 55 prefixes nor Mac acceptance was rerun.

## Reproduction

`pinned-candidate-diagnostic.tgz` changes only the wide-tile shader and its
registry identity relative to the previously retained tiled diagnostic archive.
`diagnostic-manifest.json.gz` (decompressed) replaces exactly the matching shader digest in a
separate manifest. It must be placed beside links to the unchanged model shards;
do not overwrite a deployed manifest. The binding receipt records all identities.

The initial model invocation used the old manifest with the new pinned shader.
It correctly rejected before GPU execution; `candidate-prefixes.log.gz` and
`candidate-prefixes.json.gz` retain that failure. The following pinned invocation
has zero shader mismatches and is the four-prefix numerical result above.

Decompress `operands-varied-tiles.json.gz` to a local report directory, then run:

```sh
node tests/integration/f16-projection-accumulation-replay.js \
  reports/local/q4-accumulation-20261011/operands-varied-tiles.json \
  artifacts/q4-accumulation-repair-20261011/baseline.wgsl \
  src/gpu/kernels/fused_matmul_q4_widetile.wgsl \
  reports/local/q4-accumulation-20261011/physical-varied-tiles.json q4-widetile
```

Validation: four offline arithmetic controls; native actual-operand and varied
shape replay; four-prefix replay; frozen-prefix scope checks; matmul dispatch
and config immutability controls; package shader digests, uniform layouts and
kernel registry checks. Source-only conversion digest mirrors were regenerated;
no existing model bytes or qualified package were rebuilt.

Component: GPU kernels, tooling and tests. Intent: preserved. Boundary effects:
none; generated shader identities follow the changed implementation.

## Residual-error discriminator

A subsequent capture observes only embedding and final normalization output.
Every unmasked logit across all four prefixes equals the separate retained
uninstrumented candidate exactly. The existing forensic harness was invoked
without its full-prefix-control mode and therefore failed its own postcheck
for a missing decode control; that invocation is retained. The independent
final-projection audit binds the exact archive and manifest, supplies the already
retained four-prefix control, and verifies all unmasked values directly.
Sampler-excluded token 248055 is explicitly excluded; neither failing token is
masked. The initial audit's rejection of that serialized masked value is retained.

| Vocabulary index | Local projection error | Inherited hidden-state effect | Total |
| --- | ---: | ---: | ---: |
| 18050 | -0.0000002895 | +0.0012221243 | +0.0012218348 |
| 202605 | -0.0000003524 | -0.0010689884 | -0.0010693408 |

These independent same-operand F64 dot products rule out final projection
accumulation as the material source of these failures. They do not identify a
particular upstream cache crossing or prove the residual is unavoidable.

Existing independent complete F32 controls also remain failing: chunk recurrence
has 52/55 logits above tolerance at prefixes 2/3 (maxima 0.001289/0.001310),
and scalar recurrence has 15 at prefix 3 (maximum 0.001147). Changing the label
from an F64 to an F32 reference does not establish qualification. A declared
numerical evaluation contract still needs independent justification and explicit
approval; no captured-output replacement has been activated.
