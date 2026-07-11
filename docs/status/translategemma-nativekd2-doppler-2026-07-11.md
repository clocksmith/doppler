# NativeKD2 EN/ES Translation Student Promotion Receipt

## Decision

The model `translategemma-4b-1b-enes-q4k-ehf16-af32` is accepted into
Doppler's experimental translation-specialist tier.

This is not Tier 1 or Tier 2, not a general multilingual claim, and not a
Transformers.js parity claim. The catalog keeps `quickstart: false`,
`recommended: false`, and user-facing `Experimental` plus `EN ↔ ES only`
warnings.

## Artifact

- Source identity:
  `clocksmith/gamma-translategemma-4b-1b-enes-nativekd2@sha256:c567331d8eb9d3da65205b83f4e0af06568c47ba81a7927005c1c2e72444bc87`
- Conversion config:
  `src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json`
- RDRR size: 1,025,493,248 bytes across 16 shards
- Weight policy: Q4K projections, F16 embeddings/head, F32 compute, F16 KV
- Weight-pack identity:
  `translategemma-4b-1b-enes-q4k-ehf16-af32-wp-16acf07340bd`
- Manifest variant:
  `translategemma-4b-1b-enes-q4k-ehf16-af32-mv-b3c719ee90e7`
- Conversion contract: pass, 340 tensors, layer-pattern checks 8/8
- Integrity: every declared shard SHA-256 matched
- Hosted artifact revision: `1e2c047357f07dfe45a8effc2fa915acd73999ee`
- Hosted registry revision: `6a87b8d63fcd6b3343b38dc1548a7a7936b4a6bd`
- Hosted manifest SHA-256:
  `72d51b42a51b6b429032db5edd9f89322d108c75b21ca62a2844b35c3a7123ee`
- Remote audit: the hosted manifest matched the local manifest byte-for-byte,
  and all 20 files resolved from the pinned revision.

## Coherence Review

The deterministic smoke prompts produced:

| Direction | Input | Output |
| --- | --- | --- |
| EN to ES | `The weather is nice today.` | `El clima está agradable hoy.` |
| ES to EN | `El clima está agradable hoy.` | `The weather is nice today.` |

The 128-row constructive run contained no empty outputs and no maximum-token
stops. Experimental classification remains required because Q4K loses 2.2748
BLEU and 1.7264 chrF against the same student's BF16 checkpoint, and individual
quantized outputs can contain meaning-sensitive wording changes.

## Constructive WMT13 Evidence

| Artifact/lane | BLEU | chrF | EN-ES BLEU | ES-EN BLEU |
| --- | ---: | ---: | ---: | ---: |
| Source BF16 checkpoint | 34.1896 | 59.9388 | 32.5398 | 33.9087 |
| Doppler Q4K baseline | 31.9149 | 58.2124 | 30.6799 | 31.1268 |
| Doppler Q4K optimized | 31.9149 | 58.2124 | 30.6799 | 31.1268 |

- Optimized and baseline Doppler outputs match exactly on 128/128 rows.
- Both lanes repeat exact token hashes on 8/8 determinism rows.
- The optimized lane improves median decode throughput from 31.90 to 43.69
  tok/s, median prefill from 138.28 to 198.77 tok/s, and median total latency
  from 1,023.94 to 743.19 ms on the recorded Radeon 8060S Vulkan host.
- The optimized profile is
  `profiles/translategemma-1b-nativekd2-q4k-rdna3-throughput-probe` and must not
  be generalized to Apple Metal.

The canonical receipt is
`benchmarks/vendors/results/translategemma-nativekd2/quality-wmt13/translategemma-nativekd2-wmt13-enes-128_20260711T012704Z.receipt.json`.

## Hosted Browser/WebGPU Evidence

The pinned hosted revision passed a fresh HTTP-backed Chromium/WebGPU benchmark
on Radeon 8060S / RDNA3. The optimized lane used retained Q4K materialization,
wide-tile Q4K prefill, packed-QKV split/QK-norm/RoPE fusion, and both exact-safe
RMSNorm-pair fusions.

- warmup/timed runs: 2/3
- output: `Y eso no es todo.`
- output token hash:
  `sha256:a28ace811a8bc937b278aac5261ea7b33c17473f226f7a63430f47c50601d9e9`
- median decode: 39.47 tok/s
- average prefill: 159.81 tok/s
- median prefill latency: 101.6 ms
- median decode latency: 152.0 ms
- execution contract: pass, 2/2 checks
- loader accounting: 1,025,493,248 bytes, 16/16 shards

The durable receipt is
`benchmarks/vendors/results/translategemma-nativekd2/hosted-browser/translategemma-4b-1b-enes-q4k-ehf16-af32_20260711T115839.json`.

## Teacher Comparison Boundary

The frozen demo uses the cataloged 4B teacher's fixed-corpus score of 33.6973
BLEU / 60.8011 chrF and the Doppler Q4K student's 31.9149 BLEU / 58.2124 chrF.
Both use the same 128-row WMT13 corpus and deterministic greedy decoding, but
they come from separate runtime receipts. Therefore:

- student delta: -1.7824 BLEU and -2.5888 chrF
- artifact-size reduction: 67.6%
- valid claim: a substantially smaller experimental EN/ES browser artifact
- invalid claim: teacher quality parity or a paired runtime comparison

## Demo Contract

The demo resolves this exact model ID as its preferred translation student and
shows the frozen evidence bundle by default. The hosted manifest, tokenizer
assets, and all 16 shards resolve together; browser/WebGPU translation and
catalog validation receipts close the publication gate.

The active model-card selector now admits both `text` and `translate` catalog
lanes while the shared lane resolver retains its text-only default for other
callers. A rendered Chromium smoke showed the card with `Experimental` and
`EN ↔ ES only` badges, the scoped warning, and no page or console errors.
