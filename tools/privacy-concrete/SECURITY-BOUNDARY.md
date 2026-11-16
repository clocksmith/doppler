# Research security boundary

This directory is an internal research client/evaluator, not an available Doppler or Reploid cryptographic backend. `assertExecutionBackendAvailable()` rejects cryptographic descriptors. Completing a functional fixture does not change that gate.

## Parties and computation

The trusted client holds secret keys, prompt/output tokens, embeddings, attention and nonlinear operations, quantization metadata, KV caches and sampling. The evaluator receives public model weights, public evaluation material and encrypted linear inputs. Subsequent autoregressive inputs are encrypted again on the client; there is no plaintext-token feedback message to the evaluator. The complete Qwen candidate covers all 169 linear modules, including its vocabulary head.

A future remote deployment must tolerate every remote operator colluding without obtaining the client key. That is a qualification target, **not an established security property of this implementation**. The current process harness runs client and evaluator on one host. Separate interpreters test dataflow and protocol behavior; they do not isolate the client's memory from that host's operator, root, a debugger or another sufficiently privileged process. GitHub Actions is the trusted test-client host and uses a fixed public prompt.

The client also performs every encrypted multiplication again. Only an exact serialized result from that trusted deterministic replay reaches the native result deserializer/decryptor. This rejects response substitutions tested by the harness, but gives no client compute saving, succinct proof or independently reviewed malicious-security guarantee. A malformed, missing, reordered or rejected response terminates the attempt. No noise-budget or decryption diagnostic is returned to the evaluator.

## Leakage and state

The evaluator sees public model/projection/parameter identities, the complete fixed operator/tile order, eight prefill rows followed by three single-row decode steps, public evaluation keys, frame/ciphertext lengths, session/sequence identifiers and timing. Host, process and transport metadata are also visible. Timing is not padded; it can depend on client preparation, attention and execution. The client model and its weights are public; model confidentiality is not a goal.

Each run creates fresh keys, a fresh session and fresh request-local KV caches. Reusable int8 caches contain public model weights only. Secret-reference release and process cleanup do not establish physical zeroization. Cancellation is cooperative at the documented boundaries; it cannot immediately interrupt an active native call.

The process pipe protocol binds model, plan, key, session, sequence and tile identities. Its hashes are content bindings, not endpoint authentication or signatures. There is no qualified authenticated separate-machine transport or participant identity appraisal for this research backend.

## Unfinished qualification

Production admission must remain closed until the actual deployment establishes: cryptographic parameter and implementation review; an explicit adversary/leakage contract bound into its admitted route; authenticated endpoint/key/participant binding; replay/proof soundness and selective-failure analysis; malformed-input and side-channel review; request-state isolation and cancellation under real concurrency; and model-wide correctness, memory, wire and latency evidence. Current dependency pins are not a transitive hash lock or a reproducible native-binary audit.

The pinned [native CPU implementation](https://github.com/zama-ai/concrete-ml-extensions/blob/6d3fdd1fab324874e3a2fc4a42cb3a708d5a80b4/rust/src/lib_python.rs) computes per-column encrypted dot products and compresses their outputs. Its [correctness tests](https://github.com/zama-ai/concrete-ml-extensions/blob/6d3fdd1fab324874e3a2fc4a42cb3a708d5a80b4/tests/test_matmul.py) compare approximate high-order result bits. Tiling addresses the tested arithmetic-width limit; it does not establish exact full-model arithmetic, useful throughput or a new security level. Flash's unimplemented operators and private sparse/expert access remain separate blockers.
