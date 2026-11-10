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

## Layer 25–47 bisection on the 2304-token input

A bounded suffix capture on `jupiter` (Layers 25–47, positions 1717, 2271 and
2303) paired with CPU reference checkpoints from the F32-KV routing trace
locates the exact onset of the 13-logit `quantized-v1` violation at position
2303 ([suffix-bisection-20261109.json](suffix-bisection-20261109.json)):

1. **Layers 0–24:** 0 routed-expert differences across all 2,304 tokens
   (`pos 2303` `mlp_hc.branchInput` `maxAbs = 7.272e-6`, `relRms = 3.281e-6`
   entering Layer 25 MoE).
2. **Layer 25 MoE:** `pos 2303` itself never flips a routed expert at any layer
   0–47 (`sameSet = true`, `sameOrder = true` across all 48 layers). Earlier
   tokens `pos 1717` (`1.91e-6` tie, experts `493 <-> 330`) and `pos 2271`
   (`9.54e-7` tie, experts `311 <-> 411`) flip their 10th routed expert at
   Layer 25, shifting their MoE outputs by `0.1128` and `0.1208`.
3. **Layer 26 (`linear_attention` Gated-Delta-Net recurrence):** couples the
   `pos 1717` and `pos 2271` state shift into `pos 2303`, increasing `pos 2303`
   `mlp_hc.branchInput` `maxAbs` 97.5x in one layer (`7.272e-6 -> 7.091e-4`) and
   cascading earlier-token routing flips (`2` at L25–L26 -> `12` at L32 -> `25`
   at L35 -> `37` at L39 -> `63` at L40 -> `79` at L46 -> `66` at L47).
4. **Layers 27–47 `quantized-v1` (`0.05 + 0.05 * |ref|`) gate progression at `pos 2303`:**

| Checkpoint | 4-Stream `maxAbs` (`pos 2303`) | Per-Stream `maxAbs` `[s0, s1, s2, s3]` | Direct Root-Mixer + `lm_head` (`maxAbs` / Violations / `max_excess`) | Grafted Final-State `ref_47 + d4s(L)` (`maxAbs` / Violations / `max_excess`) |
| --- | ---: | --- | --- | --- |
| `after L27` | `0.000058` | `[5.8e-5, 1.5e-5, 2e-6, 1.7e-5]` | `0.001226` / `0` / `-0.049469` | `0.000065` / `0` / `-0.049970` |
| `after L31` | `0.000215` | `[0.000215, 5.7e-5, 3e-6, 5.4e-5]` | `0.002840` / `0` / `-0.048536` | `0.000186` / `0` / `-0.049953` |
| `after L35` | `0.002280` | `[0.002280, 0.000826, 3e-6, 0.000152]` | `0.013506` / `0` / `-0.039920` | `0.001174` / `0` / `-0.049402` |
| `after L39` | `0.006537` | `[0.006537, 0.002394, 0.002263, 0.001240]` | `0.037869` / `0` / `-0.023011` | `0.010859` / `0` / `-0.042917` |
| `after L43` | `0.008704` | `[0.008704, 0.005970, 0.003386, 0.006022]` | `0.040213` / `0` / `-0.016826` | `0.051617` / `0` / `-0.027731` |
| `after L44` | `0.019919` | `[0.019919, 0.006084, 0.003858, 0.007565]` | `0.063060` / `1` / `+0.010207` | `0.045018` / `0` / `-0.015263` |
| `after L45` | `0.020158` | `[0.020158, 0.010688, 0.003933, 0.007547]` | `0.069313` / `0` / `-0.002098` | `0.062585` / `0` / `-0.012120` |
| `after L46` | `0.033456` | `[0.021718, 0.033456, 0.004943, 0.011335]` | `0.081357` / `1` / `+0.003039` | `0.071325` / `1` / `+0.001106` |
| `after L47` | `0.035573` | `[0.021457, 0.035573, 0.006362, 0.011253]` | `0.109036` / `13` / `+0.027245` | `0.109036` / `13` / `+0.027245` |

Through Layer 43, `pos 2303` has 0 violations under both direct and grafted
projections. Layer 44 (`linear_attention`, `attn_gate[0] = 1.244`, adding
`0.016758` to Stream 0) first crosses `quantized-v1` under direct projection
(1 violation). Layer 46 (`linear_attention`, `attn_gate[1] = 1.590`,
`mlp_gate = [0.036, 1.324, 1.723, 1.769]`, jumping Stream 1 `maxAbs` from
`0.010688` to `0.033456`) first crosses `quantized-v1` under grafted
final-state projection (1 violation, token `23825`). Layer 47 (`full_attention`
+ final MoE, `mlp_gate = [0.813, 1.887, 1.193, 1.921]`) combines with the
accumulated L44–L46 Stream 0/1 drift to push all 13 final violating tokens
(`[2832, 3277, 6814, 15482, 71942, 72018, 96693, 97706, 104951, 111198, 112552, 131008, 230664]`)
over tolerance.

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
