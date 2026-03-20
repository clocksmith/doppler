# Doppler Lean Module

Purpose:
- formalize high-value execution-contract invariants for Doppler manifests
- catch incompatible manifest/session/execution-v0 combinations before runtime

Current scope:
- `Doppler/Model.lean`
  - core execution-contract and kernel-path vocabulary (`KVLayout`, `Phase`, `Dtype`, `OpClass`)
- `Doppler/ExecutionContract.lean`
  - compatibility predicates for KV layout vs attention phases
  - session-consistency predicates for BDPA and tiered layouts
  - proofs for the concrete bug class where a manifest declares a decode-only KV layout while also declaring prefill attention
- `Doppler/ExecutionContractFixtures.lean`
  - `TranslateGemma`-style conflicting and fixed fixtures
- `Doppler/ExecutionRules.lean`
  - formal model of the `decodeRecorderEnabled` and `batchDecodeEnabled` rule tables
  - proofs that the happy-path contexts enable the feature and the main guard conditions force it off
- `Doppler/ExecutionRulesFixtures.lean`
  - witness checks for recorder and batch-decode decision-table behavior
- `Doppler/LayerPattern.lean`
  - formal model of alternating and every_n layer semantics
  - offset normalization and full/sliding attention assignment checks
- `Doppler/LayerPatternFixtures.lean`
  - witness checks for alternating parity, every_n stride behavior, and negative offset normalization
- `Doppler/ExecutionV0Contract.lean`
  - exact-one kernel profile pinning predicate
  - precision precedence predicate for step -> kernel profile -> session default
- `Doppler/ExecutionV0ContractFixtures.lean`
  - witness checks for precedence ordering and duplicate/missing profile rejection
- `Doppler/ExecutionV0Graph.lean`
  - ordered slot-graph model for per-phase execution
  - carried-slot compatibility check for prefill -> decode boundaries
- `Doppler/ExecutionV0GraphFixtures.lean`
  - witness checks for missing-slot rejection and phase-boundary dtype compatibility
- `Doppler/KernelPath.lean`
  - alias-resolution predicates for kernel-path registries
  - fallback activation-dtype monotonicity predicates
  - proofs that self-cycles and missing alias targets are rejected, and that valid fallback pairs never narrow precision
- `Doppler/KernelPathFixtures.lean`
  - registry and fallback-pair fixtures for pass/fail checks
- `Doppler/MergeSemantics.lean`
  - formal model of the three merge operators Doppler relies on for config behavior
  - nullish fallthrough (`??`)
  - defined overlay (`!== undefined`)
  - spread/object-field override and nullish subtree replacement
- `Doppler/MergeSemanticsFixtures.lean`
  - witness checks for null override, missing fallthrough, and subtree replacement
- `Doppler/Quantization.lean`
  - Q4K/Q6K/Q8 constant model plus padding/block-count invariants
- `Doppler/QuantizationFixtures.lean`
  - witness checks for padding monotonicity, alignment, idempotence, and block coverage
- `Doppler/RequiredInferenceFields.lean`
  - abstract distinction between non-nullable required fields and nullable-required fields
- `Doppler/RequiredInferenceFieldsFixtures.lean`
  - witness checks for missing/null acceptance and rejection semantics
- `Doppler/Check.lean`
  - simple checker entry point that renders execution-contract, kernel-path, and merge-semantics fixture verdicts

Initial target bug class:
- execution-v0 manifest contains attention steps with `phase = prefill` or `phase = both`
- session defaults choose `kvcache.layout = bdpa`
- runtime later fails because BDPA attention is decode-only

Current kernel-path target bug class:
- alias registry entries introduce cycles or missing alias targets
- finiteness fallback kernel-path mappings narrow activation dtype (`f32 -> f16`)
- runtime would silently degrade precision or fail later when those policies are consumed

Current execution-v0 target bug class:
- a step kernelRef resolves to zero or multiple kernel profiles instead of exactly one
- precision or KV I/O selection drifts from `step -> kernel profile -> session default`
- runtime would silently pick the wrong dtype or last-wins a duplicate profile

Current execution-v0 graph target bug class:
- a step reads a slot before any producer has written it
- prefill leaves a carried slot with one dtype and decode consumes it as another without an explicit cast
- runtime execution order is structurally valid JSON but semantically invalid

Current merge-semantics target bug class:
- a field that should ignore `null` is accidentally implemented with spread or defined-overlay semantics
- a field that should preserve explicit `null` is accidentally implemented with `??`
- a subtree that is intentionally replaced wholesale (`session.kvcache`, `session.decodeLoop`) is accidentally assumed to deep-merge

Current execution-rules target bug class:
- `decodeRecorderEnabled` drifts from its intended predicate around device/debug/batching/KV layout
- `batchDecodeEnabled` drifts from its intended predicate around batch size, GPU sampling, batching, BDPA-paged layout, and finiteness fallback
- rule JSON remains syntactically valid but changes command/runtime behavior silently

Current layer-pattern target bug class:
- alternating/every_n decision tables drift from runtime expectations
- negative offset normalization changes which layers become global/full-attention
- manifests remain structurally valid but apply the wrong attention regime per layer

Current quantization target bug class:
- JS constant drift breaks Q4K/Q6K/Q8 sizing between loader, converter, and kernel setup
- padding logic stops being aligned, monotone, or idempotent

Current required-field target bug class:
- nullable-required fields accidentally become null-forbidden or undefined-tolerant
- non-nullable required fields accidentally permit null/missing values
- incomplete manifests make it deeper into runtime parsing before failing

This module is intentionally narrow. It does not yet model:
- runtime merge algebra for JSON objects
- precision precedence and cast-step requirements
- manifest extraction into JSON artifacts

Manual check:

```bash
./lean/check.sh
```

Manifest-backed check:

```bash
node tools/lean-execution-contract.js \
  --manifest models/local/translategemma-4b-it-wq4k-ef16-hf16/manifest.json
```

Batch manifest sweep:

```bash
node tools/lean-execution-contract-sweep.js --root models
```

Conversion-config-backed sweep:

```bash
node tools/lean-execution-contract-config-sweep.js \
  --config-root src/config/conversion \
  --fixture-map tests/fixtures/lean-execution-contract-fixtures.json \
  --manifest-root models
```

Aggregate contract check with Lean sweeps:

```bash
node tools/check-contract-artifacts.js --with-lean
```

The manifest-backed checker:
- extracts execution steps and session facts from `manifest.inference`
- resolves missing decode-loop fields through Doppler runtime defaults
- generates a temporary Lean module under `lean/.generated-*`
- compiles it against `Doppler/ExecutionContract.lean`

The sweep runner:
- walks `manifest.json` files under the chosen root
- skips non-transformer manifests that are outside the current execution-contract scope
- runs the same Lean-backed execution-contract check on every applicable manifest
- returns a summary suitable for CI gating

The conversion-config-backed sweep:
- walks conversion config JSON files
- matches them to existing converted manifest fixtures by `output.modelBaseId`
- applies explicit fixture overrides from `tests/fixtures/lean-execution-contract-fixtures.json`
- honors explicit exclusions from the same fixture map so uncovered template/out-of-scope configs are intentional instead of accidental
- re-materializes resolved `manifest.inference` through Doppler's real conversion-plan code
- runs the same Lean execution-contract check on that materialized manifest
- keeps the contract centered on resolved manifests instead of creating a separate config-only proof model

The aggregate contract runner:
- keeps the fast JS contract checks as the default lane
- optionally folds in the manifest and config Lean sweeps with `--with-lean`
- gives CI a single command surface for the current contract families

Strict CI usage:
- `npm run ci:lean:execution-contract:configs` now requires every conversion config to be either:
  - matched to a committed fixture manifest, or
  - explicitly excluded in `tests/fixtures/lean-execution-contract-fixtures.json`
- `npm run ci:contracts:check` applies the same strict coverage rule when Lean sweeps are enabled

The same execution-contract class is now also enforced in JS manifest validation:
- [src/formats/rdrr/validation.js](/home/x/deco/doppler/src/formats/rdrr/validation.js)
- [src/config/execution-contract-check.js](/home/x/deco/doppler/src/config/execution-contract-check.js)

That means `parseManifest()` and manifest refresh flows fail fast on the same BDPA/prefill and session-default incompatibilities, even when Lean is not installed.

Expected output:

```text
["translategemma_conflicting_steps: fail", "translategemma_conflicting_session: fail",
 "translategemma_fixed_steps: pass", "translategemma_fixed_session: pass",
 "execution_rules_decode_recorder_happy_path: pass", "execution_rules_decode_recorder_rejects_bdpa_paged: fail",
 "execution_rules_batch_decode_happy_path: pass", "execution_rules_batch_decode_rejects_batch_one: fail",
 "execution_rules_batch_decode_rejects_finiteness_fallback: fail",
 "layer_pattern_alternating_even_even_layers_full: pass", "layer_pattern_alternating_even_odd_layers_sliding: pass",
 "layer_pattern_every_n_stride_layers_full: pass", "layer_pattern_every_n_non_stride_layers_sliding: pass",
 "layer_pattern_offset_normalizes_negative: pass",
 "executionv0_step_precision_precedes_profile: pass", "executionv0_profile_precision_precedes_session: pass",
 "executionv0_session_precision_used_when_no_overrides: pass", "executionv0_pinned_profile_exact_once: pass",
 "executionv0_pinned_profile_rejects_missing: pass", "executionv0_pinned_profile_rejects_duplicates: pass",
 "executionv0_graph_producer_then_consumer_valid: pass", "executionv0_graph_missing_slot_rejected: fail",
 "executionv0_graph_phase_boundary_match_accepts: pass", "executionv0_graph_phase_boundary_mismatch_rejected: fail",
 "kernelpath_valid_aliases: pass", "kernelpath_conflicting_aliases: fail",
 "kernelpath_valid_fallback_pairs: pass", "kernelpath_conflicting_fallback_pairs: fail",
 "merge_nullish_null_falls_through: pass", "merge_nullish_missing_falls_through: pass",
 "merge_overlay_null_overrides: pass", "merge_overlay_missing_falls_through: pass",
 "merge_spread_null_overrides: pass", "merge_subtree_override_replaces_base: pass",
 "merge_subtree_null_falls_through: pass", "quantization_pad_ge_input: pass",
 "quantization_pad_aligns_to_qk_k: pass", "quantization_pad_is_idempotent: pass",
 "quantization_block_count_covers_input: pass", "required_fields_nonnullable_missing_rejected: pass",
 "required_fields_nonnullable_null_rejected: pass", "required_fields_nullable_missing_rejected: pass",
 "required_fields_nullable_null_accepted: pass"]
lean-check: ok (leanprover/lean4:v4.16.0)
```

Example manifest-backed output:

```text
lean-execution-contract: wrote lean/.generated-XXXXXX/translategemma_4b_it_wq4k_ef16_hf16_ExecutionContractCheck.lean
"executionContractModule:translategemma_4b_it_wq4k_ef16_hf16_ExecutionContractCheck"
"executionContractOverall:pass"
["translategemma-4b-it-wq4k-ef16-hf16.steps: pass", "translategemma-4b-it-wq4k-ef16-hf16.session: pass"]
lean-execution-contract: ok (leanprover/lean4:v4.16.0)
```
