# All-linear Qwen backend candidate — qualification paused

This implementation covers the complete Qwen2.5-0.5B-Instruct linear route: all 24 decoder layers and the vocabulary head, 169 linear modules in total. It remains an internal candidate. The full encrypted run has **not** completed, no full-model output/cost qualification exists, and production availability remains false.

`full_model.py` owns source validation, global dynamic eight-bit quantization, public int8 weight-cache construction, model orchestration and client keys. `full_plan.py` binds source, parameters, every operator, public weights and input/output tiles into an immutable plan. `full_provider.py` processes that exact tile schedule with native Concrete GLWE. `full_bridge.py` checks the session/key/model/plan/tile/order binding and repeats each encrypted tile before any result decryption. The plan includes 387 encrypted products per model step, 1,548 across four steps. Unknown, reordered, corrupt or differently shaped model plans fail admission.

Input products wider than the conservative single-dot range are split into at most 2,048 inputs; vocabulary/output products are split into at most 2,048 outputs. Quantization and scales are computed across the whole projection. Only the resulting integer products are tiled, summed on the client and corrected/dequantized once. A native regression reconstructs a 4,864-wide raw dot outside the old signed range using three admitted tiles. This is a numerical test, not a new cryptographic security-level claim.

The client owns embeddings, all nonlinearities/attention, quantization/dequantization, bias, KV caches, sampling and **full encrypted replay**. Only matrix products are outsourced homomorphically. The driver uses a fixed eight-position prefill bucket, then one encrypted row per decode step; three independent client KV caches support floating-point, full-quantized and encrypted reference comparisons. Private prompt input uses client stdin, never provider arguments. Public int8 weight caches contain no request state or secret keys.

The [pause record](evidence/full-qwen/paused-status.json) is the source of current evidence: the real public route trace covered 169 modules; four pure plan checks and one native wide-dot regression passed. The encrypted attempt was stopped at an owned computation/wait boundary when A1 was explicitly prioritized. No generated token result is claimed. CPU client/evaluator processes were stopped, preparation/cache writes were complete, GPU was unused and Jupiter's public provider was not changed.

After A1 is resolved, the complete path can be resumed as a fresh attempt:

```sh
printf 'The capital of France is' | \
  env TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 RAYON_NUM_THREADS=2 \
  /tmp/doppler-concrete-env/bin/python tools/privacy-concrete/full_model.py \
    --model-dir /tmp/qwen05b-source --cache-dir /tmp/qwen-public-int8-cache
```

Preparation-only mode (`--prepare-only`) validates/traces the real model without keys or private input. The cache is reusable public model material; each actual run creates fresh client keys and an evaluator. Full-model execution may be expensive on this CPU-only environment. CUDA availability was not detected through `nvidia-smi` on the checked Linux fleet hosts; no remote execution or source transfer was attempted. The old denied Jupiter source-transfer action remains blocked, not bypassed.

Production Doppler/Reploid cryptographic admission remains closed. A matching backend descriptor is returned only as unqualified metadata after successful research execution; a descriptor does not enable the production availability gate. Malicious-security qualification, authenticated separate-machine transport, implementation/side-channel review and full failure testing are unfinished.

## Routing privacy and an alternative to full replay

The reviewed research does not provide a drop-in qualified replacement for this path. [CryptoMoE](https://arxiv.org/html/2511.01197v3) explicitly assumes honest-but-curious parties; its routing/dispatch work is relevant but cannot satisfy a malicious guarantee by configuration. Flash also introduces sparse indexing, n-gram access and recurrent state that this dense-Qwen implementation does not handle.

[Slalom's authors](https://github.com/ftramer/slalom) explicitly warn that the released performance prototype is unsafe for real data; its original implementation requires older TensorFlow/CUDA/SGX components. [VERITAS](https://github.com/ldsec/veritas) provides published verifiable-HE encodings over Lattigo BFV under a malicious-but-rational setting, not a reviewed adapter for this GLWE/autoregressive protocol. [Verifiable FHE](https://arxiv.org/abs/2301.07041) explains why integrity failures can become confidentiality attacks. No one of these has been substituted for replay or represented as a production malicious-security implementation. A qualified proof/verifier or independently reviewed protocol integration remains necessary.
