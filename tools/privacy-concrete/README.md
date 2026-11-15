# Encrypted transformer process experiment

This quarantined CPU experiment runs real cryptographic operations from pinned Concrete ML 1.9.0 and concrete-ml-extensions 0.1.9. It uses the public `sshleifer/tiny-gpt2` checkpoint at revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`. This is a two-layer, two-hidden-dimension test checkpoint, not the full GPT-2 model or useful language-model quality. It does not add a supported Doppler model family or enable a production privacy profile. No independent cryptographic review has occurred.

The client owns embeddings, attention, normalization, activation functions, final vocabulary projection, quantization/dequantization, masks, token selection and secret keys. A fresh evaluator interpreter receives the eight public projection matrices, a public compression/evaluation key and serialized encrypted matrices. It evaluates all eight projections for each of four autoregressive steps. Each new client activation is freshly encrypted. Tokens, clear activations, masks, scales, zero points and decrypted results are never evaluator inputs. There is no provider KV cache and no plaintext feedback. This is hybrid encrypted inference, not a claim that every transformer operation runs remotely under encryption.

## Protocol and integrity boundary

`policy.json` pins the model artifact, dependency versions, library parameter set, eight-bit quantization, fixed four-token generation, four-position context and projection geometry. Every binary frame binds its schema/policy/model/session/key/call/public-weight identities. Frames use exact header allowlists, reject duplicate fields and boolean sequence coercion, and enforce size/deadline limits. A request has one starting token and always runs four steps; EOS does not shorten its public schedule. The client checks cancellation while waiting, kills its own evaluator on failure, returns no partial output, and makes no retry or plaintext fallback.

For this tiny checkpoint, each projection is represented as a 16-by-16 matrix. The experiment explicitly uses the pinned lower-level GLWE executor and quantized-module builder, which support these concrete serialized operations; it does not alter `HybridFHEModel`'s 512-wide performance heuristic. Real input coordinates occupy the leading columns; client padding is zero and extra outputs are discarded. Symmetric public coefficients in unused positions make the upstream quantization contract applicable. A public numerical probe checks each transformed projection before cryptographic execution. These matrices and their identities are fixed independently of the private seed.

Before decrypting any evaluator result, the client repeats the entire deterministic encrypted matrix multiplication using the original ciphertext, pinned public matrix and public evaluation key. It compares exact serialized ciphertext bytes and rejects any mismatch before response deserialization/decryption. This is expensive redundant encrypted computation, not a succinct proof or an outsourcing saving. It protects this experiment's accepted-response boundary and permits direct chosen-ciphertext negative tests. It is not an independently reviewed malicious-secure FHE protocol; parser safety, implementation side channels, evaluation-key/circular-security assumptions, numerical failure probabilities, composition and realistic active-adversary qualification still need expert analysis. All remote operators may collude in the experiment's intended input-confidentiality setting; no noncolluding remote party is required. No production claim for that adversary model follows from the tests.

Public leakage is the model and fixed circuit, schedule, dimensions, public key material, ciphertext lengths, timing and process/transport metadata. Timing is observable and has not been shown independent of private data. Weights are public. The current pipe transport does not qualify separate-machine authentication, network deployment, attestation, accelerator appraisal, model-weight confidentiality, erasure or availability. The library's lossy GLWE compression also means numerical equality must be measured against the quantized model; exact ciphertext replay alone does not prove exact arithmetic or original-model equivalence.

## Run

Use an isolated Python 3.12 environment; this does not change Doppler's JavaScript runtime dependencies:

```sh
python3.12 -m venv /tmp/doppler-concrete-env
/tmp/doppler-concrete-env/bin/pip install -r tools/privacy-concrete/requirements.txt
/tmp/doppler-concrete-env/bin/python tools/privacy-concrete/prepare.py --destination /tmp/tiny-gpt2-fixture
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 RAYON_NUM_THREADS=2 \
  /tmp/doppler-concrete-env/bin/python tools/privacy-concrete/experiment.py --model-dir /tmp/tiny-gpt2-fixture
```

On the tested macOS arm64 Python 3.12 environment, `onnxoptimizer` 0.3.13 builds from source. CMake 4 rejects its old minimum policy; the build's reported compatibility option is `CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5`. This changes build compatibility, not FHE parameters. Other environments are unqualified. `requirements.txt` pins the security-relevant direct dependencies; it is not a transitive hash lock. Preparation downloads only the explicitly pinned public checkpoint/tokenizer. Inference requires local safetensors and configuration artifacts with their exact pinned SHA-256 values and uses no network download.

Dependency-free protocol tests:

```sh
python3 -m unittest discover -s tools/privacy-concrete -p test_protocol.py -v
```

Actual native ciphertext transport, replay, mutation, cancellation and disappearing-evaluator tests:

```sh
DOPPLER_CONCRETE_NATIVE=1 OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
  /tmp/doppler-concrete-env/bin/python -m unittest discover -s tools/privacy-concrete -p test_native.py -v
```

The client-local JSON report contains the generated token IDs, reference comparisons, actual traffic, setup/step/replay/round-trip time and process peak memory. Keep such output on the client; it is not a provider-safe receipt. Reported overhead belongs to this padded tiny checkpoint and this replay policy. It cannot be substituted for full GPT-2, Llama, GPU or network performance.

## Actual checkpoint tests and retained evidence

```sh
DOPPLER_CONCRETE_NATIVE=1 DOPPLER_CONCRETE_MODEL_DIR=/tmp/tiny-gpt2-fixture \
DOPPLER_CONCRETE_REPORT_DIR=/tmp/concrete-evidence \
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 RAYON_NUM_THREADS=2 \
  /tmp/doppler-concrete-env/bin/python -m unittest discover -s tools/privacy-concrete -p 'test_*.py' -v
```

[Retained measurement context](evidence/context.json) and the [Hello-token fixture](evidence/tiny-gpt2-seed-15496.json) / [The-token fixture](evidence/tiny-gpt2-seed-464.json) record actual four-step runs on macOS arm64 Python 3.12.8. All 15 tests passed, including both seeds, fresh key bindings, fixed byte counts, valid malicious evaluation (including zero input), cancellation, evaluator loss, pre-payload unsupported models, framing limits and key-reference release on evaluator startup failure.

The two recorded runs took about 19.2 and 23.4 seconds, including about 7.8 and 10.0 seconds of redundant client encrypted replay. Each sent 135,545,072 bytes and received 887,958 bytes; 134,217,808 sent bytes were the fresh public evaluation key. Both returned four greedy tokens equal to the quantized and floating-point references, with maximum encrypted-versus-quantized logit error below 0.000021. The client/test interpreter's process high-water was approximately 1 GB and child high-water approximately 510 MB; these include earlier tests, not incremental request memory. These are single functional observations with source digests, not latency distributions or full GPT-2/GPU/network benchmarks. Explicit key-reference release is not a claim of physical memory zeroization.

## Primary implementation references

- [Concrete ML hybrid inference documentation](https://docs.zama.org/concrete-ml/llms/inference).
- [Pinned 1.9.0 hybrid implementation](https://github.com/zama-ai/concrete-ml/blob/2c5f1aba337f6d4dbadf9be51a2107cb7fd13308/src/concrete/ml/torch/hybrid_model.py).
- [Pinned 1.9.0 GLWE executor](https://github.com/zama-ai/concrete-ml/blob/2c5f1aba337f6d4dbadf9be51a2107cb7fd13308/src/concrete/ml/quantization/linear_op_glwe_backend.py).
- [Concrete ML extensions](https://github.com/zama-ai/concrete-ml-extensions).
- [SEAL's decryption/oracle and ciphertext-authenticity guidance](https://github.com/microsoft/SEAL/blob/main/SECURITY.md), relevant to the separate BFV experiment and the need to analyze encrypted feedback protocols.

The upstream optimized executor runs encryption/evaluation/decryption together in `execute` mode; its `remote` path rejects optimized linear execution. This experiment owns a small explicit client projection adapter and a serialized process bridge for the encrypted multiplication call, without claiming the documentation's optimized/GPU timings are network deployment measurements. Production matching continues to reject cryptographic descriptors until a concrete backend and its full guarantees are qualified.
