# All-linear Qwen backend candidate — isolated qualification

This implementation covers the complete Qwen2.5-0.5B-Instruct linear route: all 24 decoder layers and the vocabulary head, 169 linear modules in total. It remains an internal candidate. The full encrypted run has **not** completed, no full-model output/cost qualification exists, and production availability remains false. A dedicated GitHub-hosted Linux CPU workflow runs the fixed public fixture without using A1 measurement machines.

`full_model.py` owns source validation, global dynamic eight-bit quantization, public int8 weight-cache construction, model orchestration and client keys. `full_plan.py` binds source, parameters, every operator, public weights and input/output tiles into an immutable plan. `full_provider.py` processes that exact tile schedule with native Concrete GLWE. `full_bridge.py` checks the session/key/model/plan/tile/order binding and repeats each encrypted tile before any result decryption. The plan includes 387 encrypted products per model step, 1,548 across four steps. Unknown, reordered, corrupt or differently shaped model plans fail admission. The client also checks the entire bound projection/tile cursor, fixed prefill/decode row counts and public tile weights before ciphertext disclosure. An incomplete attempt or any rejection is terminal; retries cannot skip or replace an admitted tile.

Input products wider than the conservative single-dot range are split into at most 2,048 inputs; vocabulary/output products are split into at most 2,048 outputs. Quantization and scales are computed across the whole projection. Only the resulting integer products are tiled, summed on the client and corrected/dequantized once. A native regression reconstructs a 4,864-wide raw dot outside the old signed range using three admitted tiles. This is a numerical test, not a new cryptographic security-level claim.

The client owns embeddings, all nonlinearities/attention, quantization/dequantization, bias, KV caches, sampling and **full encrypted replay**. Only matrix products are outsourced homomorphically. The driver uses a fixed eight-position prefill bucket, then one encrypted row per decode step; three independent client KV caches support floating-point, full-quantized and encrypted reference comparisons. Private prompt input uses client stdin, never provider arguments. Public int8 weight caches contain no request state or secret keys.

The [pause record](evidence/full-qwen/paused-status.json) is the source of current evidence: the real public route trace covered 169 modules; four pure plan checks and one native wide-dot regression passed. The encrypted attempt was stopped at an owned computation/wait boundary when A1 was explicitly prioritized. No generated token result is claimed. CPU client/evaluator processes were stopped, preparation/cache writes were complete, GPU was unused and Jupiter's public provider was not changed.

The v2 plan records the actual implementation versions, admits the explicitly pinned CPU-only Torch distribution, declares four CPU threads and a bounded 900-second operator deadline. Other dependency variants fail before private payload. The dedicated process must declare the thread budget before native initialization.

The complete path can run as a fresh attempt on an independently reserved machine:

```sh
printf 'The capital of France is' | \
  env TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4 \
  /tmp/doppler-concrete-env/bin/python tools/privacy-concrete/full_model.py \
    --model-dir /tmp/qwen05b-source --cache-dir /tmp/qwen-public-int8-cache
```

Preparation-only mode (`--prepare-only`) validates/traces the real model without keys or private input. The cache is reusable public model material; each actual run creates fresh client keys and an evaluator. Full-model execution may be expensive on this CPU-only environment. CUDA availability was not detected through `nvidia-smi` on the checked Linux fleet hosts; no fleet model execution or source transfer followed those checks. The old denied Jupiter source-transfer action remains blocked, not bypassed.

Production Doppler/Reploid cryptographic admission remains closed. A matching backend descriptor is returned only as unqualified metadata after successful research execution; a descriptor does not enable the production availability gate. Malicious-security qualification, authenticated separate-machine transport, implementation/side-channel review and full failure testing are unfinished.

## Routing privacy and an alternative to full replay

The reviewed research does not provide a drop-in qualified replacement for this path. [CryptoMoE](https://arxiv.org/html/2511.01197v3) explicitly assumes honest-but-curious parties; its routing/dispatch work is relevant but cannot satisfy a malicious guarantee by configuration. Flash also introduces sparse indexing, n-gram access and recurrent state that this dense-Qwen implementation does not handle.

[Slalom's authors](https://github.com/ftramer/slalom) explicitly warn that the released performance prototype is unsafe for real data; its original implementation requires older TensorFlow/CUDA/SGX components. [VERITAS](https://github.com/ldsec/veritas) provides published verifiable-HE encodings over Lattigo BFV under a malicious-but-rational setting, not a reviewed adapter for this GLWE/autoregressive protocol. [Verifiable FHE](https://arxiv.org/abs/2301.07041) explains why integrity failures can become confidentiality attacks. No one of these has been substituted for replay or represented as a production malicious-security implementation. A qualified proof/verifier or independently reviewed protocol integration remains necessary.

## Isolated full-model workflow

`.github/workflows/encrypted-qwen-full.yml` provisions only the exact public model and a fixed public prompt on a GitHub-hosted Ubuntu CPU runner. It runs focused plan/width checks, then one four-step all-linear execution. It has a six-hour job bound and a separate non-cancelling concurrency group, so ordinary pushes neither restart nor cancel it. Trigger it manually, or explicitly include `[encrypted-qwen-e2e]` in a scoped privacy commit. Other pushes skip this expensive job.

Only a constructed public-fixture evidence file is uploaded. Raw logs, key material, model caches and request state are excluded. The normal driver still accepts private client stdin locally; the CI entrypoint instead selects a fixed public fixture and does not read private input. CI is a trusted test-client environment, not a claim that a hosting operator cannot inspect its own runner. No real private user data belongs in this workflow, and hosted validation does not qualify authenticated provider deployment or malicious security.

The first isolated workflow attempt [43792487119](https://github.com/clocksmith/doppler/actions/runs/43792487119) stopped during bootstrap: a broad pip metadata check rejected an optional XGBoost wheel before model execution. It completed zero encrypted tiles and zero tokens. Bootstrap now checks imports and exact versions of the actual pinned GLWE/Torch implementation, followed by the native width regression. This change does not relax model/protocol/dependency admission.
