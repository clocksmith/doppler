# TSIR boundary-probe fixture capture (cross-repo)

This is the Doppler-side reference for capturing rung-5 frozen-Doppler-reference fixtures consumed by the Doe Cerebras-lane evidence trail. The fixtures pin per-token activations at the four TSIR boundary points (`post_rmsnorm`, `post_qkv`, `post_attn`, `post_ffn`) at L=0 from a deterministic Doppler reference inference run; the Doe rung-5 builder + validator + rung-7 oracle bind to these `.npy` files by sha256.

The fixture writer at [`src/inference/pipelines/text/tsir-fixture-writer.js`](../src/inference/pipelines/text/tsir-fixture-writer.js) is fully model-agnostic. The stage→TSIR boundary map already covers all the stages emitted by both standard attention (Gemma 4 31B, Qwen 3.6 27B full-attention layers) and linear-attention (Qwen 3.6 27B Mamba-style layers via `linear_qkv_proj`). No per-model code change is required to capture a new model — only the canonical invocation differs.

## Canonical invocations

### Gemma 4 31B

Bound by Doe at [`bench/fixtures/r3-1-31b-doppler-frozen/`](../../doe/bench/fixtures/r3-1-31b-doppler-frozen/) (cross-repo path) with `fixtureDigest=8cc17070fedf9c3dd6571714b85a96ee1715519425c0e686990909c60c80ea87`.

```sh
node tools/run-program-bundle-reference.js \
  --manifest models/local/gemma-4-31b-it-text-q4k-ehf16-af32/manifest.json \
  --model-id gemma-4-31b-it-text-q4k-ehf16-af32 \
  --prompt "The color of the sky is" \
  --tsir-fixture-dir /home/x/deco/doe/bench/fixtures/r3-1-31b-doppler-frozen \
  --tsir-fixture-layer-filter 0
```

Greedy decode produces "blue". The fixture is bound by digest in the Doe receipt at `bench/out/r3-1-31b-multi-token-decode/receipt.json` and validated by `bench/tools/validate_frozen_doppler_reference.py`.

### Qwen 3.6 27B

Target Doe path: `bench/fixtures/r3-2-27b-doppler-frozen/` (parallel to the Gemma fixture, gated by the Doe-side `bench/tests/test_validate_frozen_qwen_3_6_doppler_reference.py` which currently skips with a typed pointer to this doc).

```sh
node tools/run-program-bundle-reference.js \
  --manifest models/local/qwen-3-6-27b-q4k-ehaf16/manifest.json \
  --model-id qwen-3-6-27b-q4k-ehaf16 \
  --prompt "The color of the sky is" \
  --tsir-fixture-dir /home/x/deco/doe/bench/fixtures/r3-2-27b-doppler-frozen \
  --tsir-fixture-layer-filter 0
```

`--tsir-fixture-layer-filter 0` captures only L=0 (the four-probe set the Doe-side rung-5 expectation requires). Layer 0 in Qwen 3.6 27B is a standard-attention layer (carries `self_attn.{q,k,v,o}_proj` plus `q_norm`, `k_norm` for queryKeyNorm); the standard-attention probe-emit sites at `attention/run.js` and `layer.js` cover all four boundaries.

Once captured, the Doe-side test transitions from skip to verifying schema, artifact hashes, recomputed fixture digest, recognized Qwen modelId, and the four-probe coverage at L=0.

## Stage→TSIR boundary mapping

The fixture writer's `STAGE_TO_TSIR` map covers every stage Doppler currently emits:

| Doppler stage | TSIR boundary | Emit site |
|---|---|---|
| `post_input_norm` | `post_rmsnorm` | `attention/run.js`, `attention/record.js` |
| `linear_qkv_proj` | `post_qkv` (fused) | `linear-attention.js` (Qwen linear-attn layers) |
| `q_proj`+`k_proj`+`v_proj` | `post_qkv` (split, synthesized) | `attention/run.js`, `attention/record.js` |
| `q_norm`, `k_norm` | `post_qnorm`, `post_knorm` (debug) | `attention/run.js` (queryKeyNorm path) |
| `post_attn` | `post_attn` | `layer.js` |
| `layer_out` | `post_ffn` | `layer.js`, `ffn/standard.js`, `ffn/sandwich.js` |

For split-q/k/v models (Gemma 4 31B, Qwen 3.6 27B), the writer auto-synthesizes `post_qkv.npy` from the three projection partials by concatenating `Q ∥ K ∥ V` along the feature axis — this is the schema-binding probe the Doe rung-5 builder consumes.

## Hybrid-architecture caveat (Qwen 3.6 27B)

Qwen 3.6 27B is a hybrid full + linear-attention transformer. Linear-attention layers (Mamba-style SSM with `conv1d`) emit `linear_qkv_proj` → `post_qkv` but do not produce meaningful `post_attn` or `post_ffn` boundaries in the same shape as standard-attention layers. The L=0 capture above targets a standard-attention layer; downstream rungs (rung-7 oracle, byte-identity test) bind to that. Linear-attention coverage is a typed-blocker on the Doe side (see `docs/status/cerebras-csl.md` in the Doe repo) and not a precondition for the rung-5 fixture.

## Hardware envelope

Qwen 3.6 27B inference is a 27B-parameter run; capture requires a Doppler-runtime-supported GPU/host pairing capable of holding the model in VRAM (q4k-ehaf16 is ~16 GB on disk; runtime expansion includes activation buffers). On hardware where full inference is not feasible, the Doe-side trio still pins compile-attempt and 1L==64L byte-identity properties without the fixture (skip-when-absent test contract).

## Cross-repo references

- Doe rung ladder: `/home/x/deco/doe/docs/cerebras-north-star.md`
- Doe Qwen evidence packet: `/home/x/deco/doe/docs/cerebras-27b-qwen-evidence.md`
- Doe Gemma evidence packet: `/home/x/deco/doe/docs/cerebras-31b-evidence.md`
- Doe fixture schema: `/home/x/deco/doe/config/doe-frozen-doppler-reference.schema.json`
- Doe validator: `/home/x/deco/doe/bench/tools/validate_frozen_doppler_reference.py`
- Doe Qwen validator-binding test: `/home/x/deco/doe/bench/tests/test_validate_frozen_qwen_3_6_doppler_reference.py`
