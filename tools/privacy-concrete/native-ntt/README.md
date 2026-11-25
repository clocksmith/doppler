# Exact native CPU arithmetic research backend

This opt-in source variant preserves the original CPU APIs and cryptographic
parameter values. Separate `cpu_prepare_ntt_key` and
`cpu_matrix_multiplication_ntt` APIs use pinned `tfhe-ntt` 0.7.1 for arithmetic
modulo 2^64. It remains outside production privacy admission.

The polynomial dot product uses the original reversed clear polynomial and LWE
extraction. Packing distributes each output's original signed decomposition
digits into a polynomial, multiplies those polynomials by the original public
packing-key polynomials, then applies the original compression. Finite CRT groups
have an explicit magnitude bound below half the modulus product; reconstructed
groups are accumulated with u64 wrapping. No encryption noise, decomposition,
modulus, plaintext representation or output compression is changed.

Only public evaluation-key transforms are cached, inside the owning key and
excluded from serialization. Encrypted row transforms are request-local.
Unsupported geometry or parameter layouts reject. No global request or secret-key
cache is introduced.

The [retained native run](../evidence/native-ntt/context.json) at revision
`04ed459671632b23bcf0bee1cb8e555f6282feed` passed eight complete serialized-result
comparisons against the original arithmetic and a second fast replay, covering
partial/full packing blocks, multiple rows/input blocks and both declared
parameter sets. One default-parameter 1 x 2048 by 2048 fixture measured 8,395 ms
original versus 552 ms NTT (15.2 times faster in that fixture). The 1 x 896 by 128
fixture measured 520 versus 356 ms. Small outputs were slower: 1 x 16 by 2 measured
10 versus 347 ms. These are single native observations, not a model benchmark.
Default/high-precision key precomputation measured 173/285 ms; serialized keys
remain 134,217,808/268,435,536 bytes. Client lifetime peak RSS was 2,093,043,712
bytes across all fixtures and keys.

The isolated build verifies exact upstream files, Rust and NTT source checksums,
and the complete retained `Cargo.lock` before compilation. The build identity
binds the source and policy. The subsequent full-Qwen candidate explicitly selects
`qwen-all-linear-ntt`, with a separate immutable protocol/plan and actual installed
source/binary identities. It rejects mismatches before key disclosure; there is
no automatic substitution or downgrade. Each new model build repeats native byte
comparisons first. The original stock-library Qwen profile remains explicit.
Full-model execution and its results are documented in [FULL-QWEN.md](../FULL-QWEN.md).

This experiment is not independent cryptographic review and does not establish
malicious security, side-channel safety, memory erasure, a qualified network
backend or full Flash inference. Full client replay remains required.

Sources: pinned [Concrete ML extensions](https://github.com/zama-ai/concrete-ml-extensions/tree/6d3fdd1fab324874e3a2fc4a42cb3a708d5a80b4)
and [tfhe-ntt 0.7.1](https://crates.io/crates/tfhe-ntt/0.7.1), with exact source-file,
crate and lock hashes in `build-policy.json`. Both identify their license as
BSD-3-Clause-Clear. No native wheel, key or ciphertext fixture is published; only
source, measurements and the public dependency lock are retained.

`test_integrity.py` additionally uses actual encrypted inputs and a valid but wrong
native evaluation to exercise the shared bridge rejection boundary. It checks
wrong/corrupt ciphertexts, cancellation during real replay, terminal retry denial,
public-weight corruption before disclosure, and parser ordering for accepted
responses. These are bounded in-process verification tests; they bypass the
full-model constructor and do not qualify model admission, transport, malicious
security or cryptographic confidentiality.

> **Removed scripts (2026-11-25).** Doppler is JavaScript-only, so `Cargo.lock`, `test_integrity.py` were removed from the tree. They live at commit [`3e485ca93`](https://github.com/clocksmith/doppler/tree/3e485ca9303ee51131884b6242ce4a7fe4753c24/tools/privacy-concrete/native-ntt); the recorded receipts here are unchanged.
