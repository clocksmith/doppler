# Flash-Next: retained source-order controls

Both new candidates fail the unchanged long-input gate. No canonical runtime,
recipe, artifact, dtype default or acceptance policy changed. These experiments
extend the [F32-KV investigation](../flash-next-qualification-20261107/README.md);
they do not replace the original F16-KV failure or qualify Flash-Next.

## Prefix and local controls

A bounded capture observes positions 1717, 2271 and 2303 through layer 25,
stopping before that layer's MoE. It retains 570 tensors, 35,394,432 bytes.
Every comparable layer-25 HC slice matches the earlier parent capture bit for
bit. The [comparison](prefix-relative.json) binds the independent CPU trace,
artifact, inputs and parent observation, and reports reference RMS alongside
absolute error. Across these sampled router inputs, RMS error / source RMS
grows from approximately 1.07e-6 at layer 0 to 3.30e-6 at layer 25. This does
not cover every position or establish that all operators are correct.

The [square-product control](norm-product-control.json) simulated materializing
F32 squares before the existing reduction. It did not improve the retained
same-input norm controls. No norm change was made.

Two isolated activation candidates preserve the source's exp → addition →
reciprocal order instead of the overflow-safe negative-input rearrangement.
Outside the existing refined exponential range they retain the original stable
formulation. Independent Torch 2.11 CPU oracles use the original 2304-row batch
geometry for captured values. Deterministic range controls include extreme
values and the refined-range boundary.

| Same-input AMD control | Captured values | Exact against Torch: baseline → candidate | RMS difference: baseline → candidate |
| --- | ---: | ---: | ---: |
| Injection gate | 624 | 328 → 575 | 1.71e-8 → 6.23e-9 |
| Stream mixing | 1,597,440 | 1,200,266 → 1,528,245 | 1.09e-8 → 6.65e-9 |

Each also checks 4096 range inputs. Original-capture replay is bit-exact;
immediate and recorded results agree; both implementations remain inside the
independent F64-derived operator bound. There are no GPU or cleanup errors.
These are local numerical observations, not speed or model-quality results.
See [injection baseline](injection-baseline.json),
[injection candidate](injection-candidate.json), [mix baseline](mix-baseline.json)
and [mix candidate](mix-candidate.json).

## Full prefill result

The frozen learned-weight harness executes all 48 layers on the physical AMD
GPU, using the same artifact and independent F32-KV reference. The existing
comparator checks all 248,320 last-position logits for each input under
`quantized-v1` (absolute 0.05 plus relative 0.05). No reference selections or
captured tensors feed execution.

| Candidate | Short input: max error / violations | 2304-token input: max error / violations |
| --- | --- | --- |
| Injection source order | 2.48e-5 / 0 | 0.109030 / 13 |
| Injection and mix source order | 2.10e-5 / 0 | 0.109012 / 13 |

Both predicted first tokens agree and both long-input top-20 sets overlap
20/20. Those diagnostics do not override the failed numerical gate. The
preceding subgroup F32-KV baseline was 0.109021 / 13; neither change repairs
that failure. See [injection verdict](injection-prefill.json) and
[combined verdict](mix-injection-prefill.json). Independent decode parity,
Metal candidate coverage, model quality and performance are not established.

## Identities and reproduction

[receipt.json](receipt.json) binds every retained file. The compressed run
configurations contain the exact expanded program, kernel identities and
generator identity. Compressed observations retain source-file hashes, input
and artifact identities, numerical payload hashes, actual KV dtype and cleanup
results. Large raw tensors and logits remain operator-local; they are not
included in this public packet.

The two `.patch.gz` files are unapplied experimental diffs against the pinned
baseline shader, not production repairs. To repeat an experiment, restore the
measured source identified by the run, decompress and apply its patch in an isolated snapshot,
regenerate kernel digests with `tools/sync-kernel-ref-digests.js`, and use that
run's explicit kernel path with
`tests/integration/flash-next-learned-physical.test.js --require-model`.
Update operator-local file paths only, preserving model, input, reference,
source and execution identities. Use the unchanged
[F32 comparator](../flash-next-qualification-20261107/compare-f32kv-reference.py)
and its original reference binding. Both short and long rows must be present;
the verdict covers prefill only.

The new processes finished and unloaded. The existing public 27B provider kept
its original process identity; the original TTM limit remained unchanged. No
participant restart, kernel promotion, catalog update or deployment occurred.

Component: `doppler` (qualification observations). Intent: preserved.
Acceptance evidence: bound prefix comparison, physical same-input controls and
the two retained failing full-prefill verdicts. Boundary effects: none at
runtime; experimental diffs and retained observations only.
