# Inactive intermediate-precision proposal: memory review

Component: `doppler.repository-tooling`. Intent: preserved. This is a calculation and source review, not a runtime change, approval, numerical qualification, or new measurement. The [proposal](inactive-precision-proposal.json) remains inactive.

The broad proposal is not a proven minimal chain. Its independent ideal control is a sufficient intervention for one retained 51-prefix request, not proof that paired WGSL implementations will meet the approved F64 target. Preserve the reference, 0.001, weights, F16 KV and 1,420,000,000-byte allowance.

## Concrete calculation design

The [inventory](memory-inventory.json) freezes model geometry and source hashes. The [calculator](calculate-memory.js) accounts for a prospective **separate F32 residual sidecar** for every listed unique global tensor allocation. Existing high buffers stay allocated. This layout assumption is explicit and unimplemented; interleaved allocation could have different bucket costs. It does not append zero lows to rounded values: each paired producer and consumer would need actual compensated arithmetic.

Qwen3.5-0.8B has hidden width1,024, FFN width3,584,24layers,18linear/6full. Each12-layer partition has9linear/3full. Linear Q/K/V widths are2,048 each; full attention has8 queryheads/2 KVheads of256, and64 rotary components. The source manifest identity is retained separately from the historical physical runtime manifest identity; geometry agrees, but these are not claimed to be the same runtime configuration.

Current resident execution token-chunks the1,588-token prompt into six256-row chunks plus52 rows, and settles the recorder after every prefill layer. Decode records all12layers for one token before settling. The calculation conservatively keeps every unique listed buffer until that submission settles, plus incoming/carry/output/readback/final-normalization reserves, and charges two scratch sets for two pending attempts. It does not multiply every prefill allocation by all12layers or all7token chunks: those scopes are explicitly settled in `partition-execution.js`. A policy or lifecycle change invalidates this bound.

| Additional broad-proposal buffers, per executor | Logical or bucketed bytes |
|---|---:|
| Convolution history lows, per attempt:9×6,144×4×4 |884,736 logical |
| Same9allocations with current pool buckets |1,179,648 allocated |
| Convolution history lows for two attempts |2,359,296 allocated |
| Shared cosine and sine lows:2×8,192×32×4 |2,097,152 allocated |
| Maximum256-row prefill scratch, two attempts |76,612,096 allocated |
| Final52-row chunk scratch, two attempts |17,056,256 allocated |
| One-token decode scratch across12layers, two attempts |2,385,408 allocated |
| Scalar/frequency setup reserve |512 allocated |
| **Active broad-proposal envelope** |**81,069,056** |
| Additional idle pool envelope over inventoried sizes/usages |164,102,144 |
| **Active plus inventoried idle pool** |**245,171,200** |

The last column is GPUBuffer accounting, not process RAM or physical GPU residency. Scratch is phase-exclusive: use the maximum, not the sum. The result JSON includes every buffer width, logical byte count, bucket cost and arithmetic.

The already-approved paired recurrent state is separate:16×128×128 scalars/layer, with one additional4-byte component, costs9,437,184 extra bytes per partition/attempt or18,874,368 for two attempts. That cost is not counted twice as part of this broader proposal. Compared with historical all-F32 state, the combined conditional expansion is264,045,568 bytes including the idle envelope.

## Aliases and storage boundaries

- Prefill raw recurrent readout and normalized/gated result share one output allocation. Decode's raw readout and activated convolution remain inside the fused kernel. They are not extra global buffers.
- Prefill has separate FFN gate/up projections. Fused decode does not. The activated product still needs its low component in both phases.
- Q/K head-normalization outputs are subsequently rotated in place. There are no additional rotated-Q or rotated-K allocations in this calculation. Split query/gate outputs are separate from the original packed query/gate projection.
- Selected attention kernels keep score/softmax work in private/workgroup storage. No global token-by-token attention matrix is invented. Any new global reduction workspace must be added before implementing a candidate; register pressure/spills are not measured here.
- Paired K/V must round **directly to nearest-even F16** at the real cache write. Do not collapse to F32 first. A direct kernel writes existing F16 cache storage without another global conversion tensor. Midpoints, ties, subnormals, signs, overflow and NaNs require independent controls; zero extra global bytes does not prove correct rounding.
- Source recurrent RMS epsilon needs its high and residual low. This calculation reserves a16-byte scalar block, charged as a256-byte pooled allocation per linear-layer submission, plus512bytes for startup residual/params resources. Reusing existing uniform padding may reduce actual growth, but is not assumed. The scalar value, rounding and operation order remain part of numerical acceptance.
- Rotary lows are shared by attempts through the prepared owner/device lease. They are not allocated once per layer or attempt. They must remain charged until the final owner releases them.
- F16 KV remains unchanged. Each actual attempt reserves prompt+maximum output:1,588+4,096=5,684positions, costing34,922,496bytes per partition/attempt,69,844,992 for two. The prepared8,192-position ceiling is50,331,648 per partition/attempt. Observed1,627generated steps do not justify reserving less than the declared4,096output budget.

## The contributor boundary also needs both components

The current resident activation payload is a single F32 tensor. Retaining paired intermediates only within each contributor would discard the residual between partitions. The proposal therefore needs an explicitly declared paired activation representation in the **existing** plan, validators, payload size limits, and matching signed qualification records. This is a compatibility change, not a new transport or scheduler. It is unimplemented and must be part of any approval/acceptance scope.

For1,588×1,024 values the additional logical host/network payload is6,504,448bytes; paired payload totals13,008,896bytes. Two payloads add13,008,896 logical host bytes. Decode adds4,096bytes per transfer. Host structured clones/network buffers can multiply those bytes and are not bounded by the GPU allowance. GPU upload/readback chunk reserves are already included above. Embedding values remain exact decoded weights and final public logits remain F32.

## Historical observations are not concurrent qualification

The retained [physical allocation receipt](https://github.com/clocksmith/reploid/blob/main/artifacts/allocation-memory-20261012/summary.json) used allocation-only archive7449618b, F32 state/intermediates and sequential repetition plus cancellation/reuse. It measured809,459,080bytes on Mac input,446,967,976 on AMD output,395,753,096 on reverse Mac output and761,521,208 on reverse AMD input. It did not execute this proposal or two concurrent long attempts.

As a sensitivity calculation only, retain weights once and duplicate each historical peak's nonweight portion, then add this proposal's conditional envelope and previously approved state lows. The largest resulting sum is1,231,974,160bytes, leaving188,025,840 below1.42billion. This is **not a proven total upper bound or evidence that two attempts fit**: baseline reusable storage may be duplicated, and future workspaces, changed baseline computation, simultaneous opening and deferred destruction can add bytes. It merely shows the proposal is not excluded by the presently inventoried sidecars.

Pool accounting includes256-byte alignment, power-of-two small buckets and64KiB large increments. The idle envelope charges up to64pooled buffers, at most8per size/usage, taking the largest inventoried slots; readback has a separate usage quota. It excludes unbounded pending destruction: a released buffer can remain charged after leaving the idle pool until queue completion. Device admission must charge active, reusable and pending-cleanup bytes and reject before exceeding the unchanged cap. A stopped/unknown attempt cannot be treated as reclaimed.

## Required before claiming fit or qualification

1. Declare every new buffer, alias, precision component and cross-partition representation. Add any implementation-specific workspace to this inventory before allocating it.
2. Verify actual peak accounting on both physical placements for256/52prefill chunks, decode, two resident1,588-token attempts, cancellation while submitted, failed cleanup, retry, and settled reuse. Retain exact archive/configuration and allocation-failure receipts. No new GPU run occurred for this review.
3. Independently validate paired epsilon/nonlinear arithmetic and direct F16 rounding; run all55prefixes, both GPUs and placements using one standard package. Model usefulness and signed application acceptance remain separate.

Validation: `node artifacts/selective-materialization-20261012/calculate-memory.js --check` verifies frozen source identities, regeneration, chunk conservation, dimensional arithmetic and phase/alias exclusions. Boundary effects: none at runtime; the review identifies the future resident-activation ABI change required by the inactive proposal.
