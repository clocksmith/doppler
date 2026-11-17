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

The body-coefficient build `466b4f77f` passed CI
[44072535212](https://github.com/clocksmith/doppler/actions/runs/44072535212):
all 24 original/serial/parallel/fused/body/replay serialized comparisons, native
tamper/cancellation checks and 60 separate-process feedback calls. The same-key
35-bit fused-reference/body ratios ranged from 0.986 to 1.114, median 1.041;
five-row 2048-by-640 measured 1,628.30 versus 1,531.92 ms. This small mixed
single-observation effect does not establish a stable incremental speedup. The
probe took 84.22 seconds with a 4,559,040,512-byte cgroup peak. Its unchanged
artifacts and scope are retained as
`../evidence/storage/native-extracted-body-{arithmetic,process,context}.json`.
The weighted complete-request synthetic projection is 5.00 days on this runner;
uncontrolled hardware differences prevent comparing that total as a regression
against the prior 4.15-day observation. This remains an undeployed research
version with no production, full-model or security qualification.

The `0.1.9+ntt.5` candidate changes only packing digit materialization. Each
Rayon task owns every decomposition level and output column for eight adjacent
input masks, at most 512 KiB per block. The complete digit buffer has the same
size and layout as before. Rounding, decomposition digits, CRT groups, ordered
wrapping subtraction, compression and encrypted-row lifetime remain unchanged.
`cpu_matrix_multiplication_ntt_body_reference` retains the exact `.4` path with
serial digit materialization for same-key comparison. Both paths must match all
complete original ciphertext bytes, including partial output blocks and rows.

For three public fixtures the benchmark adds one warmup and five alternating
paired observations per path. Separate diagnostic passes report dot, digit,
packing-group and compression times; those partial sums exclude column/raw
allocation and serialization. Instrumentation is never called by the backend
and is separate from primary timings. This candidate has no measured speedup
until its exact native comparison and bounded process tests complete. The new
source/version identity requires a matching isolated build; earlier physical
receipts and any active pinned source remain unchanged.

The digit-block build `01ccabd7e` passed CI
[44085093887](https://github.com/clocksmith/doppler/actions/runs/44085093887):
all 24 complete original/baseline/candidate ciphertext comparisons, native
tamper/cancellation checks and 60 separate-process encrypted feedback calls.
Five paired observations per fixture measured median reference/candidate ratios
of 1.292 for the default 2048-by-2048 tile, 1.346 for its 35-bit/12-bit variant,
and 1.218 for five-row 2048-by-640 prefill. Every observed paired ratio exceeded
one. The separate secondary digit-materialization phases measured 76.17 versus
23.16 ms, 119.27 versus 39.10 ms, and 238.48 versus 73.36 ms, respectively.
These are public-fixture observations on one hosted runner, with raw samples
retained in `../evidence/storage/native-packing-digits-arithmetic.json`.

The bounded process probe completed in 44.14 seconds, with a 4,558,569,472-byte
cgroup peak under the eight-GiB/no-swap limit. Its logical quantizer-metadata
reservation does not include real Flash graph/model state or mappings. The
weighted native evaluator proxy remains 201,338.51 seconds (2.33 days) for the
complete four-output, 603,472-tile request, excluding client work, source/cache
I/O and transport; this is not measured full-model latency. Uncontrolled runner
differences prevent comparing that proxy against earlier totals as a speedup.
Unmodified process evidence and provenance are retained as
`../evidence/storage/native-packing-digits-{process,context}.json`. The candidate
is undeployed, and full Flash execution and security qualification remain open.

The `0.1.9+ntt.6` candidate reuses one transform scratch inside each bounded CRT
group, instead of allocating five residue buffers for every term. Scratch stays
request-local; all five arrays are overwritten before every transform. It also
specializes the initial residue conversion for the pinned decomposition digits,
including both half-base endpoints documented by
[TFHE at the locked revision](https://github.com/zama-ai/tfhe-rs/blob/1ec21a5e0b7c12165aa7e556c01e730c3117765a/tfhe/src/core_crypto/commons/math/decomposition/decomposer.rs).
For nonnegative `d`, the residue is `d`; for negative `d`, its unsigned u64 value
is `2^64+d`, so the residue is `(2^64 mod p)+d`, adding `p` if negative. The
checked bounds `|d|<=8192<p` make one correction sufficient. Values outside the
installed 10/14-bit signed domains reject. This preserves the unsigned residues,
the existing five-prime CRT bound, all transforms, rounding and compression.

Before keys, the native benchmark exhaustively compares all 17,410 admissible
signed digits against ordinary u64 modulo across all five primes, checks eight
out-of-domain values, and compares eight complete transforms while reusing
scratch across profiles, nonzero patterns and zeros. Its report schema v2 names
the new same-key `.5` allocating reference and bounded-scratch candidate
explicitly, with five paired observations and separate phase passes. Earlier
v1 receipts stay unchanged. This is an unqualified performance candidate until
its exact native byte, cancellation/tamper and bounded process checks pass;
no cryptographic or model qualification follows from the source change.

The first scratch implementation at `3e020bf22` passed exact CI
[44089615441](https://github.com/clocksmith/doppler/actions/runs/44089615441),
including all exhaustive encoding/transform checks, 24 ciphertext comparisons,
native tamper/cancellation and 60 encrypted feedback calls. Its paired medians
were mixed: about 2 to 3 percent slower for the full-width single-row fixtures,
and about 3 percent faster for prefill. It is not a general performance win.
Unmodified evidence and that disposition are retained as
`../evidence/storage/native-bounded-scratch-initial-{arithmetic,process,context}.json`.

The `.7` refinement separates the residue writes into one loop per prime after
validating the complete digit input. This permits vectorization with fixed
prime/correction values and fewer live streams; it changes no arithmetic.
Paired checks also cover the four most frequent installed Flash geometries,
derived from the pinned workload rather than selected from timing results,
alongside the prior full-width and prefill guards. New exact-build evidence is
required before making a performance claim or choosing this path as a default.

The `.7` refinement `b48dcac70` passed exact CI
[44091977449](https://github.com/clocksmith/doppler/actions/runs/44091977449),
including exhaustive digit checks, 24 complete ciphertext cases and 60 feedback
calls. It did not establish a useful speedup: the seven paired median
reference/candidate ratios ranged from 0.9981 to 1.0043. The four dominant
geometries cover 147,768 of 150,868 installed tiles per step; their paired results
were also effectively flat. Unmodified results and the rejection are retained
as `../evidence/storage/native-bounded-stream-{arithmetic,process,context}.json`.

Version `.8` therefore restores the measured `.5` allocating-transform path as
the default API. Bounded scratch remains available only through the explicitly
named `cpu_matrix_multiplication_ntt_bounded_diagnostic` and profiling APIs for
reproduction. Native checks require both the default and diagnostic to match
under the default and 35-bit domains, retain exhaustive encoding validation,
and keep tamper/cancellation tests enabled even when timing sweeps are skipped.
No new performance claim or deployed source change follows from these rejected
candidates. Security and full-model qualification remain unsatisfied;
production admission stays closed.

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
backend or full Flash inference. Full encrypted replay remains the reference
integrity mechanism; these arithmetic checks do not qualify the separate
checked-integrity research route.

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
