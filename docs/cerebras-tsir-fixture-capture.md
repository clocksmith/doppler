# TSIR boundary-probe fixture capture (cross-repo)

This is the Doppler-side reference for capturing frozen-Doppler-reference fixtures consumed by the Doe Cerebras-lane evidence trail. The fixtures pin per-token activations at the Doppler-to-CSL handoff point (`pre_layer_input`) and the TSIR boundary points (`post_rmsnorm`, `post_qkv`, `post_attn`, `post_ffn`) from a deterministic Doppler reference inference run; the Doe builder, validator, and splice receipts bind to these `.npy` files by sha256.

The fixture writer at [`src/inference/pipelines/text/tsir-fixture-writer.js`](../src/inference/pipelines/text/tsir-fixture-writer.js) is fully model-agnostic. The stage→TSIR boundary map already covers all the stages emitted by both standard attention (Gemma 4 31B, Qwen 3.6 27B standard-attention layers) and Qwen non-attention layers via `linear_qkv_proj`. No per-model code change is required to capture a new model — only the canonical invocation differs.

`.npy` fixture capture requires `--surface node`, because the writer uses the Node filesystem. Browser runs can still produce reference transcripts and coherence evidence, but they do not write Doe fixture files without a separate filesystem relay.

## Canonical invocations

### Gemma 4 31B

Bound by Doe at [`bench/fixtures/r3-1-31b-doppler-frozen/`](../../doe/bench/fixtures/r3-1-31b-doppler-frozen/) (cross-repo path) with `fixtureDigest=8cc17070fedf9c3dd6571714b85a96ee1715519425c0e686990909c60c80ea87`.

```sh
node tools/run-program-bundle-reference.js \
  --surface node \
  --manifest models/local/gemma-4-31b-it-text-q4k-ehf16-af32/manifest.json \
  --conversion-config src/config/conversion/gemma4/gemma-4-31b-it-text-q4k-ehf16-af32.json \
  --model-id gemma-4-31b-it-text-q4k-ehf16-af32 \
  --out reports/program-bundles/gemma-4-31b-it-text-q4k-ehf16-af32/program-bundle.json \
  --report-out reports/program-bundles/gemma-4-31b-it-text-q4k-ehf16-af32/reference.json \
  --prompt "The color of the sky is" \
  --tsir-fixture-dir /home/x/deco/doe/bench/fixtures/r3-1-31b-doppler-frozen \
  --tsir-fixture-layers 0
```

Greedy decode produces "blue". The fixture is bound by digest in the Doe receipt at `bench/out/r3-1-31b-multi-token-decode/receipt.json` and validated by `bench/tools/validate_frozen_doppler_reference.py`.

### Gemma 4 31B f16 lane

Target Doe path: `bench/fixtures/r3-1-31b-doppler-frozen-af16/`.

```sh
node tools/run-program-bundle-reference.js \
  --surface node \
  --manifest models/local/gemma-4-31b-it-text-q4k-ehf16-af16/manifest.json \
  --conversion-config src/config/conversion/gemma4/gemma-4-31b-it-text-q4k-ehf16-af16.json \
  --model-id gemma-4-31b-it-text-q4k-ehf16-af16 \
  --out reports/program-bundles/gemma-4-31b-it-text-q4k-ehf16-af16/program-bundle.json \
  --report-out reports/program-bundles/gemma-4-31b-it-text-q4k-ehf16-af16/reference.json \
  --prompt "The color of the sky is" \
  --tsir-fixture-dir /home/x/deco/doe/bench/fixtures/r3-1-31b-doppler-frozen-af16 \
  --tsir-fixture-layers 0
```

### Qwen 3.6 27B

Target Doe path: `bench/fixtures/r3-2-27b-doppler-frozen/` (parallel to the Gemma fixture).

```sh
node tools/run-program-bundle-reference.js \
  --surface node \
  --manifest models/local/qwen-3-6-27b-q4k-ehaf16/manifest.json \
  --conversion-config src/config/conversion/qwen3/qwen-3-6-27b-q4k-ehaf16.json \
  --model-id qwen-3-6-27b-q4k-ehaf16 \
  --out reports/program-bundles/qwen-3-6-27b-q4k-ehaf16/program-bundle.json \
  --report-out reports/program-bundles/qwen-3-6-27b-q4k-ehaf16/reference.json \
  --prompt "The color of the sky is" \
  --tsir-fixture-dir /home/x/deco/doe/bench/fixtures/r3-2-27b-doppler-frozen \
  --tsir-fixture-layers 3
```

`--tsir-fixture-layers 3` captures only L=3, the first standard-attention layer in the checked-in Qwen 3.6 27B manifest. That layer carries `self_attn.{q,k,v,o}_proj` plus `q_norm` and `k_norm` for queryKeyNorm; the standard-attention probe-emit sites at `attention/run.js` and `layer.js` cover all four boundaries.

Once captured, the Doe-side test verifies schema, artifact hashes, recomputed fixture digest, recognized Qwen modelId, and the probe coverage at L=3.

### Qwen 3.6 27B f16 lane

Target Doe path: `bench/fixtures/r3-2-27b-doppler-frozen-af16/`.

```sh
node tools/run-program-bundle-reference.js \
  --surface node \
  --manifest models/local/qwen-3-6-27b-q4k-eaf16/manifest.json \
  --conversion-config src/config/conversion/qwen3/qwen-3-6-27b-q4k-eaf16.json \
  --model-id qwen-3-6-27b-q4k-eaf16 \
  --out reports/program-bundles/qwen-3-6-27b-q4k-eaf16/program-bundle.json \
  --report-out reports/program-bundles/qwen-3-6-27b-q4k-eaf16/reference.json \
  --prompt "The color of the sky is" \
  --tsir-fixture-dir /home/x/deco/doe/bench/fixtures/r3-2-27b-doppler-frozen-af16 \
  --tsir-fixture-layers 3
```

The f16 lane is a manifest-only `weightsRef` sibling. Browser reference runs serve repo-local manifests through the repo static root so the sibling and its shared weight pack resolve under the same `/models/local/...` tree; Node fixture runs resolve the same `weightsRef` target before loading tokenizer and shards.

## Stage→TSIR boundary mapping

The fixture writer's `STAGE_TO_TSIR` map covers every stage Doppler currently emits:

| Doppler stage | TSIR boundary | Emit site |
|---|---|---|
| `layer_in` | `pre_layer_input` | `layer.js` |
| `post_input_norm` | `post_rmsnorm` | `attention/run.js`, `attention/record.js` |
| `linear_qkv_proj` | `post_qkv` (fused) | `linear-attention.js` (Qwen linear-attn layers) |
| `q_proj`+`k_proj`+`v_proj` | `post_qkv` (split, synthesized) | `attention/run.js`, `attention/record.js` |
| `q_norm`, `k_norm` | `post_qnorm`, `post_knorm` (debug) | `attention/run.js` (queryKeyNorm path) |
| `post_attn` | `post_attn` | `layer.js` |
| `layer_out` | `post_ffn` | `layer.js`, `ffn/standard.js`, `ffn/sandwich.js` |

For split-q/k/v models (Gemma 4 31B, Qwen 3.6 27B), the writer auto-synthesizes `post_qkv.npy` from the three projection partials by concatenating `Q ∥ K ∥ V` along the feature axis; this is the schema-binding probe the Doe builder consumes.

## Hybrid-architecture caveat (Qwen 3.6 27B)

Qwen 3.6 27B mixes standard-attention and non-attention layers. The non-attention layers emit `linear_qkv_proj` → `post_qkv` but do not produce `post_attn` or `post_ffn` boundaries in the same shape as standard-attention layers. The L=3 captures above target a standard-attention layer; downstream receipts bind to that layer for the frozen reference.

## Hardware envelope

Qwen 3.6 27B inference is a 27B-parameter run; capture requires a Doppler-runtime-supported GPU/host pairing capable of holding the model in VRAM (q4k-ehaf16 is ~16 GB on disk; runtime expansion includes activation buffers). On hardware where full inference is not feasible, the Doe-side trio still pins compile-attempt and 1L==64L byte-identity properties without the fixture (skip-when-absent test contract).

## Cross-repo references

- Doe Qwen evidence packet: `/home/x/deco/doe/docs/cerebras-27b-qwen-evidence.md`
- Doe Gemma evidence packet: `/home/x/deco/doe/docs/cerebras-31b-evidence.md`
- Doe fixture schema: `/home/x/deco/doe/config/doe-frozen-doppler-reference.schema.json`
- Doe validator: `/home/x/deco/doe/bench/tools/validate_frozen_doppler_reference.py`
- Doe Qwen validator-binding test: `/home/x/deco/doe/bench/tests/test_validate_frozen_qwen_3_6_doppler_reference.py`
