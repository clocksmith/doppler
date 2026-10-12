# Approved paired-intermediate implementation: operator checkpoint

The user approved the broader paired-intermediate chain after the retained memory review. This checkpoint implements explicit interleaved high/low F32 operators; it does **not** qualify a complete model. The approved independent F64 target, 0.001 tolerance, model weights, F16 KV storage, and 1,420,000,000-byte executor ceiling remain unchanged. The explicit graph is now bound and admitted for the supported paired operation set. Whole-model accuracy, concurrent memory, and useful-answer qualification remain open; unsupported paired graphs fail closed.

## Physical operator evidence

Mac Metal and AMD Vulkan independently passed the same bounded controls:

- Elementary operations: 20,570 outputs, normalized tolerance 2e-12. Maximum error 4.5474518248211915e-13 (sine). The initial range-reduction failure and an AMD receipt-write failure remain retained separately.
- RMSNorm: six width/phase/residual/weight-offset cases, source epsilon preserved as high/low uniforms. Maximum normalized error 5.808523622347618e-15.
- Projection: six F32/F16/Q4K-weight and paired/scalar-output cases. Paired maximum normalized error 2.9976021664879227e-14; scalar public outputs match independently rounded F64 result bits.
- Direct paired-to-F16 storage uses exact integer midpoint decisions. The earlier Number-reference receipts remain historical; the later exact BigInt oracle and physical public cast controls are retained in the paired-activation contract evidence. No Number(high)+Number(low) oracle is used to decide tiny-residual half ties.

Each JSON receipt binds tested source, input fixture, platform, and outcome. These are correctness controls, not performance measurements. Public idle contributors may retain unrelated GPU resources; no foreground budget or speed claim follows.

## Contracts and boundaries

A paired Tensor explicitly declares precision `two-f32`, components `2`, and elementStrideBytes `8`. Weights retain their original representation. No JavaScript tensor arithmetic or implicit widening is introduced. Final public logits can explicitly request scalar F32. Direct paired-to-half conversion avoids an intervening F32 rounding. Finite operands and finite intermediate exponent range remain required; this representation does not extend the F32 exponent range.

Immediate and recorded wrapper tests cover borrowed/owned output, malformed metadata, geometry, alias rejection, allocation failure, and failed dispatch cleanup. The generated arithmetic comes from canonical WGSL include files; the unreferenced historical duplicate was moved byte-for-byte into `historical-sources` with provenance.

Component: `doppler.gpu.kernels` and text execution configuration. Intent: preserved under the user's explicit precision extension. Boundary effects: explicit paired Tensor/partition ABI and execution-plan admission; accepted application archive unchanged.

## Paired linear-attention checkpoint

The identical convolution and gated-delta shaders pass two independent F64-equation cases on Mac Metal and AMD Vulkan: three tokens, nonzero persistent state, 128-component heads, packed and separate QKV/Z and A/B, source epsilons, and norm-weight offset. Maximum normalized output error is 9.021e-16 on Mac and 8.535e-16 on AMD; recurrent-state error is at most 1.423e-16. `linear-mac.json` and `linear-amd.json` bind exact shader and fixture hashes. The earlier Mac whitespace identity is preserved separately. Nine storage bindings are explicitly requested and checked; no default-limit assumption or speed claim is made.

Persistent convolution and recurrent layouts are now both explicit. Snapshot schema 4 preserves components through clone/restore and rejects incompatible layouts before releasing existing target state. Focused controls exercise state upload rollback, each projection failure, post-core observation failure, recorded cleanup, aliases, source epsilon packing, and allocation/dispatch failures. A reproduced pre-existing projection failure leak is repaired in the owning layer. Eight focused test files, strict implementation types, and source style passed; whole-model and paired memory qualification remain open.

The selected kernel identity now includes declared override constants when deciding weight materialization and hybrid Q4 coverage. A shared shader filename cannot authorize F32, F16, and Q4 weights simultaneously. This prevents a paired execution graph from silently changing the frozen weights.

## First model loading boundaries

Standard archives are built with `npm pack --ignore-scripts` from canonical source, with no shader substitution. `prepare-model.js` binds the exact declared paired graph while proving unchanged weights, tokenizer, source equations and frozen non-execution manifest facts. The accepted application package remains unchanged.

The retained first-model failures isolate integration defects before numerical qualification:

1. The manifest operation catalog omitted four already implemented linear projection roles; explicit catalog entries and public-parser regression repair that boundary.
2. Weight loading classified the shared paired matmul shader by filename, expanding Q4 operands. Exact registered override constants now determine weight requirements.
3. A graph mixing original F16 A/B weights and packed Q4 projections selected global mixed materialization, unnecessarily retaining dense shadows of every Q4 weight. Explicit `q4kPackedOnlyRoles` now unions requirements across all selected phases/layers and excludes any role with a dense or unresolved consumer. Those packed-only roles bypass mixed materialization; a genuinely dual-consumer role keeps both representations.
4. The first implementation of the third repair misplaced its selection variable. Its failure is retained, corrected, and protected by an actual `loadTensorToGPU` test proving a packed-only borrowed tensor creates no dense allocation. Rule-only tests were insufficient to catch that scope error.

`first-model-boundary` retains failures, including the unchanged 1.42-billion-byte rejection. Seven focused loader/materialization test files pass, as do source types, style and architecture. Archive 5 (`f3eb9b9149337f733465788033f110532efabd85d4410f81220a0c3c86d3d538`) exposed another concrete boundary: the artifact classifies Q4 tensors as generic `matmul`/`other`, while graph policy describes their semantic consumer slots. The existing layer loader now passes the bound consumer role through its loader facade to materialization without rewriting artifact metadata. Output projection aliases share the same demand union; genuinely mixed consumers remain conservative. Layer-slot and actual loader-facade regressions exercise the complete handoff. Archive 6 (`a235ce30651a7c25113e5ee718ea3a45a6d1eaf32aa5d7a205acf7b25f2c3979`) is the next installed model control, not a qualified release. Its result is tracked separately; a successful load alone will not qualify the model.
