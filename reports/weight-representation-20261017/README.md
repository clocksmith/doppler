# Per-tensor representation qualification — 2026-10-17

Status: implementation and frozen package checks pass; physical qualification is
in progress. No release promotion or performance claim. The hosted GPU-training
bootstrap failure remains separately unresolved.

The loader computes immutable requirements from each bound tensor, exact layer
and phase, explicit composite consumers, physical aliases, tied endpoints and
reachable execution-plan fallback. Unknown consumers retain conservative dense
storage. Q4 consumers can retain packed, F16 and F32 copies independently; later
legacy downcasting cannot discard a completed requirement. Preparation failure
retires allocated copies. Inference configuration and kernels are unchanged.

Baseline archive SHA-256:
`75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2`.
Candidate archive SHA-256:
`b7f5195e32c9e813f2e877c9a26ef773d067f257446a12b875d565cc5cad16f0`.
This second candidate includes the embedding/output-head alias correction.
The first candidate and initial pre-review archive are retained separately.

The generated rule bundle uses compact JSON with identical parsed rule data,
apart from the three declared per-tensor materialization selections. Readable
rule assets remain canonical. Original package limits are unchanged. Exact
archive installed-consumer smoke passes, including all 32 public export types.
Focused planner, loading, ownership, failure, alias and materialization tests pass;
strict types, architecture, style, exports and generated dependency checks pass.
These checks do not establish numerical or multi-device model qualification.

The physical driver compares the unchanged baseline with this exact candidate on
`qwen-3-5-2b-q4k-ehaf16` and `qwen3-8-27b-text-q4k-ehaf16`. The 2B runs on M3 Metal and Radeon 8060S
Vulkan; the current 27B Q4 artifact is available on AMD only. M3 27B remains
unexecuted: the artifact is not provisioned locally and available disk is below
its size. The initial obsolete Qwen3.6 manifest attempts remain failed evidence. Every served runtime file must match its archive inventory. Inputs,
context, profiles and generation settings are identical within each pair.
Cold opening means fresh browser/model state with provisioned local weights;
OS/driver caches are not purged. Recreated-device opening is separate. Both engine
orders are required. Opening, warm first token, complete response, long prefill,
model-owned GPU bytes, pool counters, aggregate process RSS, output/logit equality,
cancellation and device recreation are retained separately.

RSS is sampled every 500 ms across this browser's processes, can miss short peaks
and includes shared mappings. Pool counters are allocated GPU-buffer bytes, not
vendor VRAM counters. Hosts are not exclusive. A diagnostic driver deliberately
destroys the device after submission; this does not qualify every spontaneous loss.

Initial failures remain in raw evidence: missing snapshot provenance files, a
193-token lifecycle context used with a longer prompt, and an attempt to read the
pool after device destruction. The final driver has explicit 2,048-token context
and records absent-device state. The 27B prefill uses explicit 256-token chunks.
