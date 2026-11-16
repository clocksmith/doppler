# All-linear Qwen backend candidate — isolated qualification

[Run the demonstrated twelve-bit path](QWEN-QUICKSTART.md); the default stock profile is not the successful route.

The [research security boundary](SECURITY-BOUNDARY.md) declares parties, leakage, cohosting limits and unfinished admission gates. This implementation covers the complete Qwen2.5-0.5B-Instruct linear route: all 24 decoder layers and the vocabulary head, 169 linear modules in total. The separately bound **12-bit candidate completed four autoregressive steps** on one fixed public fixture: 1,548 encrypted tiles, with all four greedy choices matching quantized and FP32 references. Its [functional evidence](evidence/full-qwen/ntt12-four-step-success.json) records actual output and costs. Production availability remains false: this is one functional observation, not security or general model-quality qualification. The run used an isolated GitHub-hosted Linux CPU without A1 measurement machines.

`full_model.py` owns source validation, declared eight/twelve-bit dynamic quantization, public int8/int16 weight-cache construction, model orchestration and client keys. `full_plan.py` binds source, parameters, every operator, public weights and input/output tiles into an immutable plan. `full_provider.py` processes that exact tile schedule with native Concrete GLWE. `full_bridge.py` checks the session/key/model/plan/tile/order binding and repeats each encrypted tile before any result decryption. The plan includes 387 encrypted products per model step, 1,548 across four steps. Unknown, reordered, corrupt or differently shaped model plans fail admission. The client also checks the entire bound projection/tile cursor, fixed prefill/decode row counts and public tile weights before ciphertext disclosure. An incomplete attempt or any rejection is terminal; retries cannot skip or replace an admitted tile.

Input products wider than the conservative single-dot range are split into at most 2,048 inputs; vocabulary/output products are split into at most 2,048 outputs. Quantization and scales are computed across the whole projection. Only the resulting integer products are tiled, summed on the client and corrected/dequantized once. A native regression reconstructs a 4,864-wide raw dot outside the old signed range using three admitted tiles. This is a numerical test, not a new cryptographic security-level claim.

The client owns embeddings, all nonlinearities/attention, quantization/dequantization, bias, KV caches, sampling and **full encrypted replay**. Only matrix products are outsourced homomorphically. The driver uses a fixed eight-position prefill bucket, then one encrypted row per decode step; three independent client KV caches support floating-point, full-quantized and encrypted reference comparisons. Private prompt input uses client stdin, never provider arguments. Public coefficient caches contain no request state or secret keys.

The [pause record](evidence/full-qwen/paused-status.json) retains an earlier local attempt: the real public route trace covered 169 modules; four pure plan checks and one native wide-dot regression passed. The encrypted attempt was stopped at an owned computation/wait boundary when A1 was explicitly prioritized. No generated token result is claimed. CPU client/evaluator processes were stopped, preparation/cache writes were complete, GPU was unused and Jupiter's public provider was not changed.

The v2 plan records the actual implementation versions, admits the explicitly pinned CPU-only Torch distribution, declares four CPU threads and a bounded 900-second operator deadline. Other dependency variants fail before private payload. The dedicated process must declare the thread budget before native initialization.

The complete path can run as a fresh attempt on an independently reserved machine:

```sh
env TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 RAYON_NUM_THREADS=4 \
  /tmp/doppler-concrete-env/bin/python tools/privacy-concrete/full_model.py \
    --model-dir /tmp/qwen05b-source --cache-dir /tmp/qwen-public-int16-cache \
    --backend-profile qwen-all-linear-ntt12 --public-fixture qwen05b-capital
```

Preparation-only mode (`--prepare-only`) validates/traces the real model without keys or private input. The cache is reusable public model material; each actual run creates fresh client keys and an evaluator. Full-model execution may be expensive on this CPU-only environment. CUDA availability was not detected through `nvidia-smi` on the checked Linux fleet hosts; no fleet model execution or source transfer followed those checks. The old denied Jupiter source-transfer action remains blocked, not bypassed.

Production Doppler/Reploid cryptographic admission remains closed. A matching backend descriptor is returned only as unqualified metadata after successful research execution; a descriptor does not enable the production availability gate. Malicious-security qualification, authenticated separate-machine transport, implementation/side-channel review and full failure testing are unfinished.

## Routing privacy and an alternative to full replay

The reviewed research does not provide a drop-in qualified replacement for this path. [CryptoMoE](https://arxiv.org/html/2511.01197v3) explicitly assumes honest-but-curious parties; its routing/dispatch work is relevant but cannot satisfy a malicious guarantee by configuration. Flash also introduces sparse indexing, n-gram access and recurrent state that this dense-Qwen implementation does not handle.

[Slalom's authors](https://github.com/ftramer/slalom) explicitly warn that the released performance prototype is unsafe for real data; its original implementation requires older TensorFlow/CUDA/SGX components. [VERITAS](https://github.com/ldsec/veritas) provides published verifiable-HE encodings over Lattigo BFV under a malicious-but-rational setting, not a reviewed adapter for this GLWE/autoregressive protocol. [Verifiable FHE](https://arxiv.org/abs/2301.07041) explains why integrity failures can become confidentiality attacks. No one of these has been substituted for replay or represented as a production malicious-security implementation. A qualified proof/verifier or independently reviewed protocol integration remains necessary.

## Isolated full-model workflow

`.github/workflows/encrypted-qwen-full.yml` provisions only the exact public model and a fixed public prompt on a GitHub-hosted Ubuntu CPU runner. It runs focused plan/width checks, then one four-step all-linear execution. It has a six-hour job bound and a separate non-cancelling concurrency group, so ordinary pushes neither restart nor cancel it. Trigger it manually, or explicitly include `[encrypted-qwen-e2e]` in a scoped privacy commit. Other pushes skip this expensive job.

Only a constructed public-fixture evidence file is uploaded. Raw logs, key material, model caches and request state are excluded. The normal driver still accepts private client stdin locally; the CI entrypoint instead selects a fixed public fixture and does not read private input. CI is a trusted test-client environment, not a claim that a hosting operator cannot inspect its own runner. No real private user data belongs in this workflow, and hosted validation does not qualify authenticated provider deployment or malicious security.

The first isolated workflow attempt [43792487119](https://github.com/clocksmith/doppler/actions/runs/43792487119) stopped during bootstrap: a broad pip metadata check rejected an optional XGBoost wheel before model execution. It completed zero encrypted tiles and zero tokens. Bootstrap now checks imports and exact versions of the actual pinned GLWE/Torch implementation, followed by the native width regression. This change does not relax model/protocol/dependency admission.

The client `run(..., cancelled=threading.Event())` path checks cancellation before preparation, before encryption/disclosure, during pipe I/O, after evaluator/replay computation and before decryption/finalization. Once observed, cancellation permanently closes the attempt, including if the event is later cleared. A native encryption, replay or decryption call already in progress is cooperative: it can finish before the next check; immediate native interruption or physical memory zeroization is not promised. No encrypted result is deserialized after cancellation observed during evaluator or replay work.

The current client emits bounded failure codes before progress observers; exception messages stay in client-local diagnostics. Constructed public evidence accepts only known codes, validated fixed-fixture output types and complete success reports, and rejects a contradictory terminal failure. The collector is intended for the trusted fixed-fixture workflow; a fixture label supplied in an arbitrary file is not authentication or authority to declassify private output.

The [authenticated evaluator interface](SSH-DEPLOYMENT.md) binds an explicitly pinned SSH endpoint/account and complete model plan. It requires pre-provisioned files and requester authorization, installs nothing and has no local fallback. Its transport qualification is separate from full-model and cryptographic qualification.


The clear quantized reference uses signed int64 dot-product accumulation. A
float32 matmul followed by an integer cast can round wide integer sums before
comparison; a dedicated regression uses a nonrepresentable odd sum above 2^24.
This correction does not alter the encrypted circuit, parameters or verification
boundary, and does not establish full-model parity by itself. Previously retained
measurements remain scoped to their exact recorded revisions.

## Explicit exact-arithmetic candidate

`qwen-all-linear-ntt` binds a separate protocol and actual native source/binary
identities to the same complete 169-module route. The stock-library profile is
unchanged. This candidate uses the [measured exact NTT arithmetic variant](native-ntt/README.md),
without changing cryptographic parameters or removing client replay. Eight public
native cases produced byte-identical original/fast/replay ciphertexts; large
matrices improved while tiny output widths regressed. Those measurements do not
establish full-model speed, model quality or security qualification.

`.github/workflows/encrypted-qwen-ntt.yml` builds the pinned CPU source with its
retained exact dependency lock, repeats the native comparisons, admits the actual
installed identity, and executes four all-linear model steps. It has a separate
non-cancelling concurrency group and a three-hour budget; trigger only manually or
with `[qwen-ntt-e2e]`. The original stock Qwen and Flash MoE runs are neither
restarted nor cancelled. This is a changed implementation test, not a repeat of
the unchanged six-hour candidate. The default-precision variant subsequently failed its third-step token check;
its separate outcome is retained below. It is not the successful twelve-bit route. Its process-pipe
execution is not a qualified separate-machine deployment; the existing SSH
profile does not accept the new protocol.

Source-file identity is now guarded across the complete client model/tokenizer
materialization and public route trace, before any private key generation.
Same-length rewrites, replacements and restore-after-read mutations terminate
preparation. Public tile snapshots still independently bind the later int8 replay
weights. Filesystem checks do not protect against a malicious client kernel or
constitute an atomic snapshot of an adversarial filesystem.

## Separate clear quantization checks

`qwen_quantization.py` compares all 169 real-model linear modules on eight fixed
public prompts, four steps each, using the same prefill/decode helper as encrypted
execution. It records independent quantized/floating greedy sequences and
teacher-forced logit errors/cosines. Keys and encryption calls are mechanically
forbidden in this lane. The workflow is isolated CPU-only and does not repeat or
restart any existing encrypted run. These are small public quality fixtures, not a
corpus benchmark, full-model encrypted evidence or a qualification shortcut.

The [actual eight-fixture int8 check](evidence/full-qwen/quantization-eight-fixtures.json)
completed in 56.75 s with 4,701,478,912 bytes client lifetime peak RSS. Independent
four-token sequences matched FP32 on **5/8** fixtures; teacher-forced greedy choices
matched **28/32**. Japan, freezing and sky completions differed. Maximum logit error
reached 3.6714. This is a measured precision limitation, not evidence of lossless
Qwen behavior. Keys/encryption were forbidden, so this difference is independent
of the NTT evaluator. The [context](evidence/full-qwen/quantization-context.json)
binds the exact revision/run and public source.

`quantization_probe.py` adds one fixed sky-prompt boundary diagnostic from embedding
and input norm through the first Q projection, plus a **clear-only** 12-bit
full-model control. Records include bounded public slices, full tensor digests,
statistics and an explicit tolerance-policy ID. `wide_reference.py` stores wide
coefficients as int16 and accumulates products in int64; it rejects encrypted mode.
This clear-only control cannot itself enter encrypted mode or register a production path. The separately bound encrypted candidate below has independent execution gates.

## Completed attempt outcomes and explicit precision correction

The [retained attempt records](evidence/full-qwen/attempts-context.json) distinguish
arithmetic verification from model numerical correctness. The stock run exhausted
its six-hour job budget after 322 verified tiles; its first 387-tile prefill did
not finish. It was not restarted. The default-precision NTT run replay-verified
**1,161 tiles**, completing all 169 linear modules on each of three model passes.
Its third greedy-token comparison failed after two accepted steps. No partial
output was published. Exact replay passed; the model numerical acceptance gate
failed. This is not a completed four-token encrypted result.

`qwen-all-linear-ntt-48` is a distinct candidate protocol, using the already declared
48-bit/four-level research precision profile and explicit 300 MB frame limit.
The original stock/default-NTT profiles retain their identities and parameters.
A paired public first-Q probe compares the same actual inputs and weights under
both parameter sets before one complete four-step high-precision attempt. The
greedy-reference requirement is unchanged. A successful paired probe alone does
not establish why the third-step model check failed or qualify the complete model.
The new full workflow has a separate concurrency group and a four-hour budget.

The clear integer reference now accumulates bounded blocks into int64 and applies
one global correction/dequantization. This preserves the exact integer arithmetic
while avoiding a full vocabulary-head int64 temporary; geometry limits come from
`reference-policy.json`. Source-precision and encrypted profiles remain distinct.

Increasing stored precision and evaluation-key samples has no independent security
review. Unchanged encryption noise does not establish unchanged security. Production
admission remains closed even if the numerical variant completes successfully.

## Complete 12-bit encrypted candidate

`qwen-all-linear-ntt12` now wires the complete same 169-module autoregressive
route to 12-bit dynamic quantization, signed int16 public coefficient caches and
a separately declared 35-bit message domain with 48-bit storage/four-level packing.
It has its own immutable plan/protocol, encoding and parameter identities. Cache
byte strides, lengths, hashes and coefficient ranges are validated before key
export; both client and evaluator load the same explicit int16 representation.
Legacy profiles retain signed int8 caches and their original message domains.
The client still owns nonlinearities, attention, decoding and full replay.

A conservative 2,048-input raw-dot bound is 17,175,674,880, below the signed
35-bit limit of 17,179,869,184. This is an arithmetic-range check, not a probability
of decryption failure or cryptographic-security proof. A public signed-boundary
native probe precedes one complete four-step model run in
`encrypted-qwen-ntt12.yml`. The probe has an explicit empirical error criterion;
passing it alone cannot qualify model output or privacy. The separate full-model
artifact now records the completed four-step public fixture.

The wider clear control and this encrypted candidate share the pinned source
quantizer and int64 reference arithmetic. Changing the plaintext/message domain
requires new correctness and independent parameter/implementation security review.
Production cryptographic admission remains closed; the model does not silently
fall back to eight-bit or provider-visible execution.

## Retained precision evidence

The [clear twelve-bit control](evidence/full-qwen/clear-twelve-bit-control.json)
matched all eight independent four-token FP32 sequences and all 32 teacher-forced
greedy choices. Maximum teacher-forced logit error was 0.4394; these small public
fixtures do not establish corpus quality. In the eight-bit sky fixture, embeddings,
input normalization and the first Q input were exactly equal; the first Q output
introduced maximum absolute error 0.0585842. No encryption or key generation ran
in that control.

The [same-input actual encrypted first-Q probe](evidence/full-qwen/same-input-storage-precision.json)
measured integer error 73 under default storage and 1 under the separate 48-bit
storage profile; dequantized errors were 0.00231361 and 0.0000815392. The key grew
from 134,217,808 to 268,435,536 bytes. This supports the numerical correction
hypothesis on this input, without proving the earlier third-step failure cause.
The [measurement context](evidence/full-qwen/precision-context.json) binds both
records to their exact revisions and artifact hashes. Neither result qualifies
a complete encrypted model or the altered cryptographic parameters.

The [actual twelve-bit message-domain probe](evidence/full-qwen/twelve-bit-message-domain.json)
completed on the candidate's exact native build before full-model execution. At
2,048 inputs and extreme signed coefficients, the expected negative/positive raw
dots were -17,175,674,880 and 17,167,288,320; maximum measured integer error was 9.
Complete repeated ciphertext bytes matched. This is one disposable-key numerical
observation, not a cryptographic noise-failure bound or full-model result. The
complete four-step twelve-bit Qwen run passed its separate public-fixture gate;
independent security and broader model-quality gates remain open.

The [higher-storage full-model attempt](evidence/full-qwen/ntt48-numerical-failure.json)
also stopped at **1,161 verified tiles**, with `token-reference-mismatch` on its
third step. Its native replay checks passed. Reducing the same-input first-Q error
from 73 to 1 integer units did not establish complete-model parity; stored precision
alone was insufficient in this attempt. The artifact and exact revision are in
the [attempt context](evidence/full-qwen/attempts-context.json). This unchanged
candidate has not been retried. The independently declared twelve-bit candidate
retains its own full-model gate; no plaintext correction or relaxed greedy
acceptance substitutes for an encrypted result.

## Completed twelve-bit full-model functional observation

The [exact-revision run](https://github.com/clocksmith/doppler/actions/runs/43841061585)
executed every linear module across all 24 decoder layers and the vocabulary head,
then returned four tokens `[12095, 13, 1084, 374]`: “ Paris. It is”. All **1,548**
encrypted tile responses passed replay verification. All four greedy choices
matched both the quantized and FP32 references on this public prompt. Maximum
encrypted-versus-quantized logit error was **0.1350**, so matching greedy choices
is not numerical equivalence or an arbitrary-prompt accuracy guarantee.

One hosted CPU observation took **5,926,799 ms (98.8 minutes)** end to end.
Encrypted prefill took 71.16 minutes; the three decode steps took 9.16, 9.08 and
9.01 minutes. Client encrypted replay consumed 49.05 minutes and evaluator round
trips 49.14 minutes. The run sent **321,932,548 bytes**, received **86,305,332 bytes**
and used a **268,435,536-byte** evaluation key. Client lifetime peak RSS was
**5,265,190,912 bytes**. Complete quantized-reference comparisons added about
2.9 seconds; floating references also ran on the client. These costs are not a
matched production overhead benchmark or evidence of useful client offloading.

The [context](evidence/full-qwen/ntt12-functional-context.json) binds the exact
checkpoint, protocol, native source/binary, parameter profile and artifact hash.
This establishes a working real small-Qwen hybrid prototype, beyond an isolated
projection. It does not establish full Flash execution, separate-machine privacy,
independently reviewed malicious security or production admission. All normal
cryptographic capability gates remain closed.

The [reference-validation feedback analysis](SECURITY-BOUNDARY.md#observable-reference-validation-feedback)
records the historical third-pass stopping-position and component acceptance-bit
channels. Current drivers complete and close the fixed admitted route before
local numerical validation; Qwen teacher-forces the actual encrypted token history
into the same reference checks after closure. Rejection releases no output and
integrity/cancellation still fail immediately. New reports mark
`validationBoundary: after-route-close/v1`; the retained four-step result above
predates that change. Runtime/timing leakage and production security remain
unqualified, and no additional full-model performance run is implied by the fix.


The eight installed public quality prompts now also own encrypted fixture IDs in
`quantization-fixtures.json`. CLI prompt selection and safe evidence collection
share `public_qwen_fixtures.py`; an unknown fixture rejects before artifact reads,
and the selected identity must match the report before output can be published.
The legacy capital default is preserved. The separately marked arithmetic run
uses the same all-layer/twelve-bit route and four steps, testing a second prompt
and the new closed-route boundary. Its inspected result is recorded below.


## Second prompt with completed closed-route validation

The [retained arithmetic result](evidence/full-qwen/ntt12-arithmetic-closed-route.json)
completed all **24 decoders and output head**, **169 linear modules**, **1,548 verified
encrypted tiles and four autoregressive steps**. All four greedy tokens matched
both the quantized and FP32 reference. Maximum encrypted-versus-quantized logit
error was **0.12256**, so this is not numerical equivalence. For the public prompt
`2 + 2 =`, the raw continuation tokens were `[220, 19, 198, 59]` (space, `4`, newline,
backslash). Reference agreement does not turn that continuation into a general
answer-quality result.

The actual run reports **`after-route-close/v1`**: the complete fixed encrypted
route finished and closed before the reference acceptance checks. Total elapsed
time was **137.44 minutes**, including **68.38 minutes** client encrypted replay
and **68.37 minutes** evaluator round trips. Application frames sent/received were
**321,932,548 / 86,305,332 bytes**, including the **268,435,536-byte** evaluation key.
Client lifetime peak RSS was **5,243,269,120 bytes**. These are one hosted CPU
observation with process-local transport, not separate-machine measurements or a
matched production overhead benchmark.

The [context](evidence/full-qwen/ntt12-arithmetic-context.json) binds the exact
revision, artifact checksum and checkpoint, and distinguishes this real execution
from the later cache-lifecycle changes. This extends fixed-fixture functional
coverage to two prompts and exercises the closed-route validation boundary.
Client nonlinearities and full replay remain; timing/failure leakage, arbitrary
input correctness, independent malicious-security review and full Flash generation
remain unqualified. Production admission stays closed.
