# Inference Test Matrix Strategy

How Doppler tests the combinatorial space of model families, quantization formats, attention modes, batch sizes, and decode paths without maintaining a full cross-product.

## Problem

The full Cartesian product of inference dimensions is intractable:

| Dimension | Cardinalities |
| --- | --- |
| Model family | Gemma, Qwen, LFM2, TranslateGemma, ... |
| Quantization | F16, Q4K, Q6K, ... |
| Attention mode | full, sliding window, hybrid, linear, gated delta net |
| KV cache layout | standard, bdpa-paged |
| Batch size | 1, N (multi-token) |
| Decode surface | generate (streaming text), generateTokens (streaming ids), generateTokenIds (batch ids), decodeStepLogits (manual) |
| Readback interval | 1, N, null |
| Kernel path | default, fused, experimental |
| Activation dtype | f16, f32 |
| Finiteness guard | enabled, disabled |

Even with 2-3 values per dimension, the cross-product exceeds 10,000 combinations. Testing all of them is neither practical nor necessary.

## Strategy: layered coverage + representative slicing

### Layer 1: Contract and config tests (fast, no GPU)

Test that invalid combinations fail fast before reaching the GPU. These are cheap and exhaustive within their domain; run them while editing and before pushing. CI runs a subset of them.

- Manifest schema validation: required fields, nullable-required semantics
- Config merge order: profile/config precedence, override semantics, execution compile
- Kernel path resolution: registry lookup, alias chains, dtype contract
- Tensor-config consistency: norm flags match tensor presence
- Execution plan compilation: step validation, kernel ref pinning

**Location:** `tests/config/`, `tests/inference/execution-*.test.js`, `tests/converter/`

### Layer 2: Kernel and primitive tests (per-kernel correctness)

Test individual GPU kernels against CPU references. Each kernel is tested in isolation with synthetic inputs. Dimensions tested: dtype variants, workgroup sizes, edge-case shapes.

These are per-kernel, not per-model. A matmul Q4K test covers Q4K for all models.

**Location:** `tests/kernels/`, `tests/gpu/`

### Layer 3: Integration tests (representative model slicing)

Use one canonical model per architectural feature. Do not duplicate across quant formats unless quant is the feature under test.

| Feature | Representative model | Why |
| --- | --- | --- |
| Hybrid sliding/global attention | Gemma3-1B Q4K | sliding window + global layers |
| Linear attention / gated delta net | Qwen3.5-0.8B | conv layers + recurrent state |
| Conv layer state | LFM2 | conv kernel init, state management |
| Translation chat template | TranslateGemma-4B | structured message handling |
| Small F16 baseline | Gemma3-270M F16 | dtype/layout coverage |
| Q4K fused kernels | Gemma3-1B Q4K | kernel path + quantization |

**Location:** `tests/integration/`, runtime verify profiles in `src/config/runtime/experiments/verify/`

### Layer 4: Decode surface parity (structural + behavioral)

Test that all generation surfaces honor the same contract obligations without running the full GPU pipeline.

- **Structural parity test:** `tests/inference/generate-token-ids-contract.test.js` verifies that `generateTokenIds()`, `_runDecodeLoop()`, and `_generateTokensInternal()` all include the same contract checks (abort, EOS, stop sequences, finiteness handling, stats, cleanup).
- **Behavioral parity benchmark:** `tools/bench-text-decode-paths.js --teacher-forced` runs both the token-return and logits-return paths on the same teacher sequence and fails if outputs diverge.

### Layer 5: Performance benchmarks (on demand)

Cross-engine and cross-configuration benchmarks. Not part of CI.

- Vendor benchmark registry: `benchmarks/vendors/`
- Workload definitions: `benchmarks/vendors/workloads.json`
- Decode path comparison: `tools/bench-text-decode-paths.js`

## Pairwise slicing for integration

When a change touches a cross-cutting concern (e.g., KV cache layout), test the affected dimension against one representative from each other dimension, not the full product.

Example: KV cache layout change

| KV layout | Attention mode | Quant | Batch |
| --- | --- | --- | --- |
| standard | full (Gemma3) | Q4K | 1 |
| standard | linear (Qwen3.5) | Q4K | 1 |
| bdpa-paged | full (Gemma3) | Q4K | N |

This covers the risk surface (layout × attention mode, layout × batch size) without multiplying by every quant format and model family.

## Where each layer runs

Test tiers and what CI runs are defined in the
[testing runbook](testing-runbook.md#ci-notes). For this matrix: CI runs a
subset of Layer 1 contract tests; Layer 2 runs locally or through the
`Manual Runtime Validation` workflow; Layer 3 uses `npm run verify:model` on
model changes; Layers 4 and 5 run on demand (`tools/bench-text-decode-paths.js`,
`tools/vendor-bench.js`, `tools/compare-engines.js`).

## Adding a new dimension

When a new inference dimension is introduced (e.g., a new attention variant):

1. Add contract/config tests (Layer 1) for the new config field
2. Add kernel tests (Layer 2) if new shaders are involved
3. Pick or create one representative model (Layer 3)
4. Add a verify profile in `src/config/runtime/experiments/verify/`
5. Update this doc with the new representative model entry

Do not create a new test for every existing model × new dimension combination.
