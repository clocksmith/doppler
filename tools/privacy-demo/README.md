# BFV autoregressive protocol experiment

This is an **unqualified protocol experiment**, isolated from Doppler's shipped runtime. It does not enable any privacy profile or register an execution backend. Its eight-symbol bigram fixture is not a supported language model. No independent cryptographic review has been performed.

Run with Node 22 or newer:

```sh
cd tools/privacy-demo
npm ci --ignore-scripts --no-audit --no-fund
npm test
npm run demo
```

`policy.json` owns the fixed model fixture, step count, protocol and parameter policy. `node-seal` 7.0.0 is pinned with package-lock integrity; Microsoft SEAL owns cryptographic operations. BFV uses degree 4096, SEAL's `BFVDefault` coefficients at `tc128`, and a batching plaintext modulus of 20 bits. SEAL validates the parameter set. A parameter check is not a review of this protocol or its implementation.

The client generates a fresh key context per run. A separate evaluator process receives only public model/protocol metadata, Galois evaluation keys and BFV ciphertexts. It computes public transition scores homomorphically. The client decrypts the eight scores, chooses the next token locally, then freshly encrypts the next state. Four autoregressive steps run without plaintext token/score feedback to the evaluator. This demonstrates a complete toy feedback loop, not only one encrypted matrix multiplication. Weights are public, and prompt history beyond the last symbol is outside this bigram model's semantics.

The experiment assumes a semi-honest evaluator and any transport operator colluding with it. It requires no noncolluding remote server, but this does **not** qualify the requested all-remote malicious-adversary model. Evaluation-key/circular-security assumptions, ciphertext sanitization, adaptive/chosen-ciphertext attacks, client side channels and composition require expert review. A provider can replace a valid ciphertext with one containing incorrect scores, influencing subsequent encrypted feedback. The test suite demonstrates that attack explicitly; it cannot claim integrity or active-adversary confidentiality from successful decryption. Production cryptographic inference remains unavailable.

Public leakage includes model, protocol, parameter set, evaluation keys, fixed step count, ciphertext sizes, timings and process/transport metadata. The input ciphertext shape is fixed across fixture symbols. There is no TLS/session attestation, remote identity appraisal, verifiable computation, differential privacy, model-weight confidentiality, large-model quantization, transformer attention, MoE access-pattern protection or hidden variable output length. Per-step metadata binds model, protocol, full parameter/leakage policy digest, local session, evaluation-key digest and step, but it authenticates no remote operator. The pipe transport is a local process boundary.

Client and provider frames have exact allowlists and byte limits. Client secret keys and plaintext states are never serialized to the evaluator. Wire tests inspect message fields and equal sizes, exercise cancellation and corrupt/wrong-context responses, and verify ordinary reference equality. Such checks do not prove cryptographic security. Each process owns its WASM handles and closes them on success, rejection and cancellation; cleanup is not a remote-erasure claim.

A sample Apple arm64 Node 22 run produced four correct tokens in 277 ms: about 212 ms client setup, followed by 35/9/9/9 ms steps, with 2,319,325 bytes sent and 701,368 bytes received, and 19 bits minimum remaining BFV noise budget. The client reported about 86 MiB peak RSS; provider memory was not measured. The same four-step plaintext fixture took 0.023 ms in that run; this tiny CPU baseline does not predict transformer overhead. This is one functional measurement, not a benchmark against ordinary inference, nor a transformer performance prediction. Subsequent runs may differ. CLI output is client-local and includes the fixture result; it is not a provider/public receipt.

Primary dependencies: [node-seal source](https://github.com/s0l0ist/node-seal) and [Microsoft SEAL](https://github.com/microsoft/SEAL). Before enabling a real backend, supply model-specific protocol messages and validated serialization, a precise malicious/collusion/leakage model, fresh key/endpoint binding, a secure multi-step decoding design, independent review, and measurements on that actual model. No advertisement or successful toy run satisfies those requirements.
