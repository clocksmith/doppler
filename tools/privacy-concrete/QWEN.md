# Small-Qwen encrypted projection experiment

This is a concrete step toward a real-model hybrid encrypted backend, quarantined from Doppler Run. It loads the official Apache-2.0 `Qwen/Qwen2.5-0.5B-Instruct` safetensors checkpoint at revision `7ae557604adf67be50417f59c2c2f167def9a775`. The installed profile pins every loaded model/tokenizer/configuration artifact, the named projection, its rectangular geometry and quantized public weights. Inference is offline; unsupported sources/profiles fail before key generation or provider startup.

**Only the matrix product inside `model.layers.0.self_attn.k_proj` executes homomorphically.** Its matrix is 896 inputs by 128 outputs. The complete Qwen model runs to generate text, but embeddings, all other projections, nonlinearities, attention, masks, decoding and all request state remain client-owned. The client also repeats the encrypted projection before decrypting every response. `fullModelEncrypted`, `qualified` and `productionBackendAvailable` are explicitly false.

The provider receives a fresh public compression/evaluation key, the pinned public matrix and ciphertexts. It receives no prompt tokens, plaintext activations, masks, quantization scales, secret key or decoded feedback. Four autoregressive steps use a fixed 16-position context bucket, with at most 13 initial tokens. Different prompt lengths in that bucket have the same measured wire lengths. Each run has fresh client keys and a separate evaluator; there is no provider-side request-state or KV cache. Public model weights may be shared independently of private request state in a future backend.

`profiles.py` owns immutable installed contracts and arithmetic admission; `protocol.py` owns bounded framing; `bridge.py` owns the client/evaluator lifecycle and replay; `provider.py` owns encrypted evaluation; `linear.py` owns trusted client projection preparation. Remote capability claims cannot create profiles or change their exact model/weight/parameter binding. Tiny-GPT2's existing wire contract remains distinct.

## Run and check

Use the pinned Python 3.12 environment from [README.md](README.md). Explicit preparation downloads only public pinned artifacts:

```sh
/tmp/doppler-concrete-env/bin/python tools/privacy-concrete/prepare.py \
  --profile qwen05b-k-projection --destination /tmp/qwen05b-source
printf 'The capital of France is' | \
  env TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 RAYON_NUM_THREADS=2 \
  /tmp/doppler-concrete-env/bin/python tools/privacy-concrete/real_model.py --model-dir /tmp/qwen05b-source
```

Prompt input is read from client stdin; it is not put in evaluator arguments or a prompt-valued CLI option. The client-local JSON contains generated text and diagnostic comparisons: do not treat it as a provider-safe receipt.

```sh
DOPPLER_QWEN_MODEL_DIR=/tmp/qwen05b-source DOPPLER_QWEN_REPORT_DIR=/tmp/qwen-evidence \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 RAYON_NUM_THREADS=2 \
  /tmp/doppler-concrete-env/bin/python -m unittest discover -s tools/privacy-concrete -p test_real_model.py -v
/tmp/doppler-concrete-env/bin/python tools/privacy-concrete/scaling_audit.py --model-dir /tmp/qwen05b-source
```

Dependency-free framing/profile tests run in CI. Actual native crypto, tiny-transformer and Qwen tests require their explicit opt-in flags/local artifacts and perform no automatic model download.

## Measured scope

The two [source-bound Qwen reports](evidence/qwen/context.json) record four generated tokens for each public fixture. “The capital of France is” continued as “Paris. It is”; “A computer is” continued as “programmed to generate a”. Both matched quantized and floating-point greedy references at every tested step. This is a narrow correctness observation, not general model quality qualification.

The recorded CPU runs took approximately 57.2 and 64.0 seconds, including 26.3 and 29.7 seconds of full client encrypted replay. Each sent 135,784,092 bytes and received 460,400 bytes; 134,217,808 sent bytes were the fresh public evaluation key. Individual encrypted-generation steps took about 12.6–15.9 seconds; the separately measured floating-point client reference steps took about 0.10–0.18 seconds. Maximum encrypted-versus-quantized logit error was below 0.00081; eight-bit quantization introduced a larger, separately reported difference from floating point. These single observations do not establish a performance distribution or ordinary Doppler comparison.

Client/test process high-water memory reached about 4.34 GB, and evaluator high-water about 460 MB. These are process high-water marks, including preceding work in the test interpreter, not incremental request memory or a sum of concurrent processes. Model/kernel import time and source hashing are outside the reported function timing; setup includes local loading, public calibration, fresh keys and public-material upload.

## Security and practical limits

The intended confidentiality setting needs no noncolluding remote party, but this experiment does **not** establish production security against malicious colluding operators. Exact deterministic ciphertext replay rejects changed computations before response decryption. It repeats all remote encrypted work locally, provides no client-compute saving, and is not a succinct proof or independent cryptographic review. The library's encryption/evaluation-key assumptions, parser safety, side channels, lossy compression error/failure probabilities, adaptive composition and end-to-end protocol require expert analysis.

Leakage includes the public model/selected route, parameters, fixed context bucket/schedule, public evaluation material, ciphertext sizes, timing and process/transport metadata. Timing has not been shown independent of private data. Public numeric range admission is not a cryptographic security-level assessment. Native computation is not interruptible mid-call; cancellation is checked at boundaries and during blocked reads/writes, with owned evaluators terminated on failure. Key-reference cleanup is not physical memory zeroization.

The local pipe is a logical protocol boundary on a trusted client host. It does not protect against a privileged or same-user process inspecting the client machine and does not qualify authenticated separate-machine transport. No remote-provider deployment, attestation, erasure, availability, full-model encryption or backend integration is claimed. Production cryptographic descriptors remain unavailable.

## Concrete blockers to Qwen3.8 Flash

The [pinned source/tensor audit](evidence/qwen/scaling-audit.json) separates inspected geometry from measured execution. Small Qwen has 168 decoder projection weight tensors; 24 down projections exceed the current conservative signed raw-dot range. Their uint64 public matrix representation would total approximately 2.86 GB before other model components. Only the named key projection has been executed by this path.

The official [Flash-Next configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json) declares `qwen4_exp` and Transformers 5.8 development support; the pinned 4.44.2 client does not implement it. Its 2,560-wide hidden projections exceed the current conservative 27-bit signed raw-dot budget. Admission rejects that range: changing flags cannot qualify it. Tiling or reparameterization needs its own numerical, protocol and security validation.

The [publisher's Flash-Next card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/README.md) describes hybrid recurrent/sparse attention, MoE routing and n-gram embeddings. Expert choice, sparse indexing and table access patterns need a declared privacy protocol; a GLWE linear primitive does not hide them automatically. Recurrent/attention state also needs validated policy/key/cache isolation. Rounded publisher parameter counts imply roughly 360 GB at two bytes per parameter or 720 GB at four, before vision/overhead; these are storage estimates, not measured residency. Keeping a full replay model on this 24 GiB client is not feasible by the current method.

A useful next backend therefore needs more than more selected layers: qualified arithmetic for wider dots, an explicitly private access-pattern/state design, realistic model placement and an integrity protocol that offers useful client savings. Neither semi-honest assumptions nor removing replay may be advertised as production malicious security. No Flash encrypted execution or extrapolated Flash latency has been measured.

> **Removed scripts (2026-11-25).** Doppler is JavaScript-only, so `bridge.py`, `linear.py`, `native-ntt/prepare.py`, `prepare.py`, `profiles.py`, `protocol.py`, `provider.py`, `real_model.py`, `scaling_audit.py`, `test_real_model.py` were removed from the tree. They live at commit [`3e485ca93`](https://github.com/clocksmith/doppler/tree/3e485ca9303ee51131884b6242ce4a7fe4753c24/tools/privacy-concrete); the recorded receipts here are unchanged.
