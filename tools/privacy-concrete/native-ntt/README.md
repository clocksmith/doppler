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

The `0.1.9+ntt.2` candidate evaluates independent CRT packing groups on the
existing Rayon pool, then performs the same wrapping subtraction in the original
group order. Group boundaries and magnitude bounds, decomposition, moduli,
noise, key material and final compression are unchanged. Reconstructed group
buffers are request-local and bounded by the existing fixed key geometry. A
separate `cpu_matrix_multiplication_ntt_serial` diagnostic entry point provides
same-key serial/parallel byte and cost comparisons; it is not an advertised
backend or a fallback. The build identity and dependency version change, so an
old installed wheel rejects before disclosure against this source. Existing
physical readiness work must keep its explicit earlier source/build pin.

The native benchmark compares original arithmetic, serial NTT, parallel NTT and
parallel replay, including every installed Flash tile shape in the actual
35-bit/12-bit domain and a five-row representative. Serial/parallel order
alternates across public cases. Until the exact build completes those checks and
the separate-process resource/feedback probe, this is an unqualified performance
candidate. No speedup, full-model latency or security claim follows from the
parallel source change alone.

The separately versioned `0.1.9+ntt.3` candidate additionally uses the
checksum-pinned crate's `prime32::Plan::mul_accumulate` and `normalize` operations.
Within each original bounded CRT group it sums unnormalized NTT products, then
applies the same inverse-size factor once before the inverse transform. Linearity
over each prime gives the same residues as normalizing every product first.
This removes per-term product clones and scalar remainder accumulation without
changing the CRT group bound or any cryptographic parameter. The serial baseline
and `cpu_matrix_multiplication_ntt_parallel_reference` retain the old unfused
arithmetic for same-process comparisons; neither is an advertised fallback.
The benchmark alternates fused/unfused-parallel order and requires every complete
serialized result to match the original arithmetic. Native qualification and
new all-shape resource measurements are required before claiming improvement.

The parallel-only `0.1.9+ntt.2` build at `59471aed0` passed CI
[44065844984](https://github.com/clocksmith/doppler/actions/runs/44065844984):
all 24 original/serial/parallel/replay serialized comparisons, native tamper and
cancellation checks, and 60 separate-process encrypted feedback calls. In the
16 same-key 35-bit cases, serial/parallel time ratios ranged from 1.75 to 2.97
(median 2.40); the five-row 2048-by-640 fixture measured 4,521.44 versus 1,912.71
ms. These are individual synthetic observations, not a repeated-trial or
full-model speedup claim. The separate-process probe completed in 87.83 seconds
with a 4,558,827,520-byte cgroup peak under the hard eight-GiB limit. Its
unchanged artifacts and explicit scope are retained in
`../evidence/storage/native-packing-parallel-{arithmetic,process,context}.json`.
The fused candidate is separate and is not qualified by this earlier result.

The fused `0.1.9+ntt.3` build at `6d0b37504` subsequently passed CI
[44067892517](https://github.com/clocksmith/doppler/actions/runs/44067892517):
all 24 original/serial/unfused-parallel/fused/replay serialized comparisons,
native tamper/cancellation checks, and 60 separate-process encrypted feedback
calls. The 16 same-key 35-bit cases measured a median 3.36 serial/fused time
ratio and 1.33 unfused-parallel/fused ratio. Five-row 2048-by-640 measured
4,335.09 ms serial, 1,745.11 ms unfused-parallel and 1,331.26 ms fused. These
remain individual synthetic observations with no confidence interval. The
separate-process probe took 66.50 seconds with a 4,558,839,808-byte cgroup peak
under the hard eight-GiB cap. Unchanged artifacts and provenance are retained in
`../evidence/storage/native-packing-fused-{arithmetic,process,context}.json`.
This validates the observed arithmetic/resource component scope, not actual
Flash output, full-model memory, practical outsourced integrity or production.

The `0.1.9+ntt.4` candidate keeps the mask convolution and packing but computes
only the ciphertext-body coefficient used by the pinned
[upstream extractor](https://github.com/zama-ai/concrete-ml-extensions/blob/6d3fdd1fab324874e3a2fc4a42cb3a708d5a80b4/rust/src/computations.rs).
For each input block, reversing the zero-padded clear polynomial makes its
product coefficient `N-1` equal the wrapping dot product of the original body
and clear coefficients; no negacyclic wrap contributes to that coefficient.
The other body coefficients are never extracted. The candidate therefore avoids
the body forward/inverse NTT without changing the returned LWE mask or body.
Ciphertext bodies remain request-local owned buffers, with no secret-key cache
or new protocol message. A full-GLWE fused diagnostic baseline and all older
baselines remain for complete serialized-result comparisons, including partial
input blocks, multiple blocks, rows and output packing blocks. Version/source
identity changes require a new build; this candidate is unqualified until its
exact native and resource checks complete.

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
