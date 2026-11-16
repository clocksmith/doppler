# Exact native CPU arithmetic experiment

This is an opt-in research source variant, not an installed model backend or a
production privacy claim. The original CPU APIs and parameter values remain
available unchanged. New `cpu_prepare_ntt_key` and `cpu_matrix_multiplication_ntt`
APIs use pinned `tfhe-ntt` 0.7.1 for exact arithmetic modulo 2^64.

The polynomial dot product uses the original reversed clear polynomial and LWE
extraction. Packing distributes each output's original signed decomposition
digits into a polynomial, multiplies those polynomials by the original public
packing-key polynomials, then applies the original compression. Finite CRT groups
have an explicit magnitude bound below half the modulus product; reconstructed
groups are accumulated with u64 wrapping. No encryption noise, decomposition,
modulus, plaintext representation or output compression is changed by this
arithmetic variant.

Only public evaluation-key transforms are cached, inside the owning key, and
excluded from serialization. Encrypted row transforms are request-local.
Unsupported geometry/parameter layouts reject. There is no global request or
secret-key cache. Client/provider policy admission has not been extended to this
variant; replacing a library without a separately admitted profile is unsupported.

The isolated workflow verifies exact upstream files, pins Rust and NTT source,
checks the NTT crate checksum and TFHE revision in the dependency lock before
building, and compares complete serialized results with the original arithmetic.
It covers full/partial packing blocks, multiple encrypted rows and input blocks,
plus both declared parameter sets. Timings include result serialization, separate
key precomputation, and one observation per public fixture. No speedup or model
execution is claimed before those actual checks complete.

The first build generates a public Cargo lock for retention; a generated lock
and matching function outputs are not independent cryptographic review. This
variant remains outside production admission and does not establish malicious
security, side-channel safety, memory erasure or useful model throughput.

Sources: pinned [Concrete ML extensions](https://github.com/zama-ai/concrete-ml-extensions/tree/6d3fdd1fab324874e3a2fc4a42cb3a708d5a80b4)
and [tfhe-ntt 0.7.1](https://crates.io/crates/tfhe-ntt/0.7.1), with the exact crate
SHA-256 and source-file hashes in `build-policy.json`. Both upstream packages
identify their license as BSD-3-Clause-Clear. No native wheel is published by
this experiment; only source-constructed measurements and the public dependency
lock are retained.
