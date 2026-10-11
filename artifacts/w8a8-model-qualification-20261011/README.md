# Frozen W8A8 model qualification

The independent CPU comparison fails the frozen application-quality gate. W8A8
answers 5/12 tasks correctly, F16 5/12 and the incumbent Q4 2/12. The gate requires
10/12 and preservation of every F16-passing task. W8A8 loses `literal` while gaining
`copy-code`; all three complete every task with EOS. These failures are retained.

The separate likelihood criterion passes: over identical 200 teacher-forced
targets, W8A8 mean NLL is 4.717551487921449, F16 is 4.630961223030469 and Q4 is
5.1531543895274305. W8A8's increase over F16 is 0.08659026489098043, below the frozen
0.10 limit. This approximation limit does not replace distributed tolerance 0.001.

`source-comparison.json` binds all inputs and compressed reports. Decompress each
`.json.gz` normally; the index records the SHA-256 of the uncompressed bytes.
No standard package, model catalog, numerical fixture or deployment was changed.

## Independent computation

`tools/gemma3-deployed-generation-control.py` loads every declared deployed tensor
and checks the manifest, tokenizer and shard identities. The pinned original
Gemma source configuration/tokenizer supplies independent Transformers equations.
Arithmetic outside quantized projections is F32 and attention caches store F16.

For W8A8, `tools/lib/w8a8_model_reference.py` replaces exactly the 126 explicitly
encoded linear matrices with dynamic per-token activation quantization, exact
integer-code GEMM and two separately rounded F32 rescale operations with explicit
subnormal-result flushing. Integer GEMM uses binary64 BLAS: products and all
partial sums are integers smaller than 2^31, so no integer accuracy is lost.
This is a numerical oracle, not an integer CPU speed claim. Every replacement's
execution count is retained. It imports no converter or runtime arithmetic.

The independent BigInt reference cross-check covers random exponent scales, a
near-halfway input where intermediate F32 division produces the wrong code,
integer accumulation above 2^24, signed-zero flushing and rejected inputs.
Native WGSL floating operations do not universally guarantee nearest-even;
GPU equivalence requires the separately tested explicit rounding implementation.

The existing F16 artifact differs from nearest-even conversion of the identified
BF16 source only in 18,887 values in the F16 subnormal region, with maximum
5.960464477539063e-8. The actual incumbent remains the comparison baseline; neither
its bytes nor historical results were replaced. Both original lineage and
expected-F16 rounding audits are retained.

## Executed scope

- `node tests/kernels/w8a8-reference.test.js`: independent rational reference.
- `python3 tests/integration/w8a8-model-reference-test.py`: three cross-reference controls.
- Existing Gemma deployed-weight decoding test: five controls.
- Three CPU full-model evaluations with four threads each; every task and teacher-forced target retained.

Installed portable-model execution and lifecycle results are recorded below.
Packed-variant qualification and fixed-work timings remain separate checks.
Passing execution or operator controls does not close the failed model-quality gate.

## Installed observations and corrected lifecycle

`installed/index.json` retains every initial failed invocation, including two
harness assumptions (explicit BOS duplicated the runtime's BOS; an empty linear
state map was mistaken for active recurrent state) and one runner variable error.
It also preserves the runtime defects these controls exposed: first-token abort
reporting, incomplete deferred unload settlement, and successful CPU-logits
readback retaining an unrequested GPU buffer. The owning runtime fixes are
separate commits; thresholds and model bytes remain unchanged.

`w8a8-installed-model-05` runs the corrected archive
`79b7c68ceb23d3e156e5e6287d978bfbe218940b832c2460b80efe2472e9b44b` through its declared
`./compat`, `./tooling/device` and `./partitions` exports. All 12 tasks and 200
teacher-forced targets execute; three repeated requests, submitted cancellation,
subsequent generation and settled unload pass. Every pool ownership category is
zero after unload. The private pool import observes ownership; model work uses
the installed public handle. Execution success is **not** model qualification.

GPU task quality is 4/12 and source token equality is 7/12. GPU W8A8 NLL is
4.769074093670992, exceeding the frozen F16 +0.10 approximation limit. Earlier
same-package F16 and Q4 controls match all 12 source token sequences, with NLL
4.631039722160251 and 5.153211447267374 respectively; their original cleanup
failures remain recorded. No gain or standard package promotion is claimed.

The additional independent sequential-decode source control has NLL
4.694931130174279. It does not replace the frozen full-prefill source report.
Their phase difference does not explain the GPU mismatch: after input tokens
`[2,236776]`, both source paths produce the selected logit about 16.68065, versus
GPU 19.36719. The first BOS-only prediction agrees closely.

At this specific second-token boundary, the installed observer captures layer-0
normalization and Q/K/V without changing any of the 262,144 baseline logit bits.
`installed/actual-operand-comparison.json` independently verifies every Q/K/V
output bit against the **actual** GPU operands and immutable weight bytes.
Those projections are correct at this boundary. Independently evolved source
normalization inputs differ by at most 1.52587890625e-5; activation codes agree,
while the scale differs by one F32 step. This closes that local projection
hypothesis, not the remaining full-model disagreement.

Fixed-work benchmark code is prepared but no timing campaign or speed advantage
is established by these records. Further residual localization uses this same
frozen two-token context; it must preserve the baseline and reference separation.


## First material residual divergence

The same two-token context, exact installed archive and immutable weights were
observed again with baseline logits unchanged in every bit. All duplicate
`layer.out` observations were required to agree before comparison. Independent
source and GPU residual differences remain below 0.000733 through layer 9, then
reach 33.6621 at layer 10. The retained compact records are in `diagnostic/index.json`.

Layer 10's seven Q/K/V/O/gate/up/down projections are all bit-exact against the
independent integer and ordered-rescale calculation using their **actual GPU
operands**. Q/K/V also match independently evolved source outputs. The O input
differs from source by at most 5.37e-7, crossing one activation-code rounding
boundary; its output differs by 0.003644. The FFN gate/up operands cross one code
boundary, then the down operands differ in 97 codes. Their raw output difference
of 0.010802 becomes 33.1506 after the learned post-FFN normalization.

Independent binary64 evaluation of normalization, GELU and residual addition on
actual GPU operands differs from the corresponding GPU observations by at most
6.104e-5. These checks account for amplification through discontinuous activation
quantization and learned normalization; they establish no defective projection
or nonlinear operation at this boundary. They do not establish source equality
or excuse the failed full-model quality and approximation gates.


The separately identified packed-dot graph (`67eee682...`) produced exactly the
same 12 token sequences, visible texts and all 200 teacher-forced observations
as the portable graph on archive `79b7c68...`. This is a model-variant control,
not an improvement in quality. `installed/variant-output-comparison.json` binds
the two complete reports and the diagnostic derivation retains original source
and weight identities.

Both initial packed attempts fail the additional submission-time cancellation
control: cancellation is issued immediately after `queue.submit` returns, before
awaiting completion. The second receipt retains 13,094,144 deferred bytes in 133
resources after the public operation rejects and the queue settles; active
allocation remains unchanged. Final unload does release all resources. This
exposes a cancellation-settlement defect distinct from the passing first-token
callback cancellation. Neither failure is removed by the subsequent repair.


## Repaired submission-time cancellation

Archive `47d7d18fb51a0852b6b0c35b0590e306b460c743eefe2d36b6615839c8651edc`
contains the failed-prefill settlement repair. `w8a8-installed-model-06` and
`w8a8-packed-installed-model-03` both pass the complete model execution controls:
12 tasks, 200 teacher-forced targets, three repeated requests, first-token
callback cancellation, actual submitted-prefill cancellation, deterministic
reuse, and settled unload. Submitted cancellation rejects with `AbortError`;
active ownership stays unchanged and deferred bytes/counts are zero after the
public operation and submitted queue have settled. All ownership categories are zero after unload.

Both variants retain identical task tokens/text and teacher-forced observations,
also unchanged from the earlier candidate. `installed/model-quality-disposition.json`
keeps model quality separate: 4/12 tasks pass versus F16 5/12 and Q4 2/12; W8 loses
the F16 literal-copy pass. NLL 4.76907409 exceeds the frozen F16 +0.10 limit.
Neither variant qualifies for promotion. Earlier baseline and failed-candidate
archive identities remain explicit in that report; no historical result was
regraded or replaced.

## Final completion metadata control

Archive `7c240dad4d1ae98e204e9727e687b9bc4379abd758b16af0fd857a99d7adac8d`
corrects initial stopping classification. `w8a8-installed-model-07` and
`w8a8-packed-installed-model-04` pass complete execution and both cancellation
controls. All twelve task token sequences/texts and all 200 teacher-forced
observations remain unchanged from `47d7d18f...`; every task now reports the
public `stop-token` reason, including initial stop token 106. The earlier
`stop-token-or-sequence` metadata defect remains visible in historical receipts.
`installed/completion-and-lifecycle-final.json` records the mechanical comparison.
Peak owned GPU buffer allocation is 668,488,920 bytes for each variant; every
ownership category is zero after unload. Quality remains rejected at 4/12 and
NLL 4.76907409. This package remains a diagnostic candidate, not the standard pin.
