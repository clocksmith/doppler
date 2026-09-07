# F32 to F16 conversion rounding repair

Component: `doppler.runtime-source.converter`. Intent: preserved.
Boundary effects: newly converted tensor bytes and Q4_K scale encodings may
change. Previously converted and signed artifacts are not rewritten. No model
publication, catalog promotion, or runtime fallback is authorized by this record.

The later diagnostic repair also touches
`doppler.runtime-source.inference.pipelines.text`: configured logits-probe
forwarding and failure-path cleanup only. `doppler.repository-tooling` retains
the measured package delta. Intent is preserved in both components; the
session-order failure remains an unresolved runtime observation.

## Observed defect

An F16 precision control for Reploid's document-answer investigation sampled 16
evenly spaced values in each of 320 converted text tensors. All source/converted
shapes and the 24-layer pattern matched. Of 5,120 samples, 54 values across 14
linear-attention normalization tensors differed from source values rounded to
nearest F16, ties to even. These tensors originate as F32; the source also
contains BF16 tensors. The full sample record preserves every value, not only
failures.

The completed integrity check independently rehashed the 4,548,221,488-byte
source checkpoint and all 57 original F16 shards. The checkpoint matches its
retained SHA-256 and every shard matches its manifest BLAKE3 commitment. The
original manifest SHA-256 is
`9051eeff39378ac6bf84560ca676fc4ab24e50534fd450d99155aae20fc37734`.
`integrity.tar.gz` retains the verifier, completed report, and command output.

For example, source `0.955810546875` rounds to `0.9560546875` (`0x3ba6`), but the
converter emitted `0.95556640625` (`0x3ba5`). The old `float32ToFloat16()` discards
the lower mantissa bits. A focused regression using this exact source value
fails on the original function with `15269 !== 15270`.

The source checkpoint is `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Its acquisition record, original F16
manifest, conversion configuration, and numeric samples are retained in the
archive. The original conversion uses the installed Reploid Doppler 0.6.0
package. The repair modifies the owning Doppler source, not that installed
package and not the existing artifacts.

## Repair and checks

`src/converter/quantizer.js` now rounds nearest-even for normal and subnormal
halves, including exponent carry and overflow. Its declaration documents the
encoding rule. Tests cover signed zero, infinity, NaN, the exact failing source
value, and both signs immediately below, at, and above every adjacent finite-half
midpoint. A converter integration fixture verifies actual F16 shard words from
F32 source values rather than deriving expected words with the same encoder.

- Original regression: fails on the recorded value.
- `node tests/converter/quantizer.test.js`: passes.
- `node tests/converter/core-bf16-f16-conversion.test.js`: passes.
- `node tools/run-node-tests.js tests/converter`: 44 files pass.
- `npm run source:style:check`: passes.
- `npm run source:architecture:check`: passes.
- `npm run catscan:check`: passes, 30 components.
- `npm run typecheck:source`: passes.
- `npm run artifact:contract:check`: passes, seven of 31 catalog models exposed.
- `git diff --check`: passes.

The broader unit run completed: 781 test files passed. Installed package smoke,
model-release checks, and Program Bundle checks also passed. Installed execution
smokes use synthetic programs and are not physical qualification.

The fresh conversion completed with 57 shards and 320 text tensors. All 5,120
samples now equal the source rounded to F16, with no failing tensor samples.
Source checkpoint SHA-256 and all 57 shard BLAKE3 commitments passed independent
rehashing. The new manifest SHA-256 is
`b7b2f051c82aba757101eaa301fa6eccae91d3d56be9a52f1543e079544e131e`.
The 59-file executable inventory contains 3,772,899,642 bytes. This remains an
unsigned local precision control, not a qualified Capsule.

`verified-conversion.tar.gz` retains the completed conversion report, manifest,
config, source snapshots, lineage, samples, integrity verification, artifact
inventory, and completed check logs. It omits model weights. The output root is
`/tmp/reploid-qwen35-answer-screen.06HATl/f16-rne-model`.

The complete green workflow first failed the package size budget. The unchanged
1,780-file closure contains six changed shipped files adding 2,597 source bytes.
`package-audit.json` records the source revisions and per-file accounting. The
measured package is 2,107,701 packed bytes and 10,869,933 unpacked bytes. The
budget adjustment preserves closure, forbidden-file, file-count, and size gates;
the public boundary check then passed. A subsequent workflow failure exposed a
stale routing inventory after the Gemma qualification manifest was committed.
Its regenerated inventory records 30 manifests and preserves 32 surfaced
integrity findings. Neither regeneration nor this repair resolves those findings.
The subsequent complete main-worktree workflow passed, including 789 unit-test
files. Public CI for `f1fc6885d012b6b99022eb625d517c9583ae9698` also passed:
https://github.com/clocksmith/doppler/actions/runs/34154371398.

An isolated worktree at `/tmp/doppler-qualified-candidate.YQ9eiC` starts from
`ae153af0a235b283c9ba8b8f80a356f3c8d570d2` and applies the same measured package
budget and generated routing changes. Fresh `npm ci` and its complete
`npm run check:green` pass, including 789 unit-test files. `isolated-checks.tar.gz`
retains those terminal logs. Installation reports two high-severity development
dependency findings; the production-only audit reports zero vulnerabilities.
No dependency versions change, and passing these checks is not semantic model
qualification or proof of deployed source identity.

The first physical attempt stopped before generation because the temporary
browser context offered 1.75 GB of storage against a 3.51 GB model-cache
requirement. It generated zero cases and completed cleanup without errors. This
is an infrastructure failure, not evidence about model output quality. A fresh
persistent-profile retry retains the same model bytes, prompts, generation
settings, and semantic acceptance requirements.

## Exhaustive codec and corrected-browser observations

The independent NumPy comparison checks every one of 1,881,825,088 stored
text-tensor binary16 words. The corrected artifact has zero mismatches and no
nonfinite source or expected values. The original has 362,893 mismatches:
362,477 originate in BF16 tensors and 416 in F32 tensors. Exactly those 362,893
words differ between the artifacts. The original and corrected artifacts have
identical tensor descriptors, architecture, tokenizer metadata, tokenizer bytes,
and inference configuration. This establishes the codec correction over all
stored text tensors, not inference equivalence or useful answers.
`full-tensor-reference.tar.gz` retains the verifier, full per-tensor results,
examples, aggregate counts, and completed main-worktree green-check log.

The corrected browser attempt 02 completes all eight unchanged development
cases on Chrome 145.0.7632.6, physical Intel gen-12lp, fallback false, with no
cleanup errors. Its execution report SHA-256 is
`76b5422ed196b1c252c0639ccae34f380e4bc1ec78d3d67ecd64b3dc8878b411`.
The fresh persistent profile starts at zero storage usage; load takes
167,123.7 ms in this observation. Concurrent repository checks mean this is not
a controlled performance comparison.

Only the sensor answer passes the frozen combined requirements. The ticket
answer omits both required facts. Museum, carton, roof, team, builder, and
orchard-injection cases repeatedly emit the partial-unknown sentence until the
256-token limit. They omit required facts or conflicts and do not provide the
permitted complete abstention. These are coverage, stopping, and contract
failures; this observation does not invent unsupported factual claims.
`corrected-browser.tar.gz` preserves both the failed cache attempt 01 and
completed attempt 02, including 591 captured browser-source files for attempt
02. The result is rejected: better codec accuracy has not qualified the assistant.

Comparison with the published Q4K run alone cannot isolate rounding because
precision and execution policy also differ. The original F16 artifact is
retained as a known nonconforming adverse control, not admitted as qualified
source-rounded model output. Its paired browser run uses a fresh persistent
profile and the exact 591 frozen sources captured from corrected attempt 02.
The paired run completes all eight cases with clean teardown. Every original
token sequence and output string is exactly identical to the corrected run.
Both score 1/8 under the unchanged combined requirements. Thus the codec repair
has no observed answer-quality benefit on these exposed regression cases.
No prompt, acceptance rule, generation setting, or kernel changes are introduced
for this comparison.

`original-browser-comparison.tar.gz` retains the original run and executable
pair comparator. `f16-paired-answer-comparison-02.json` records equal browser,
device, load policy, generation controls, corpus, prompts, and all 591 frozen
browser-source hashes. The exposed runtime profiles differ only in local server
address, manifest hash, and bound session identity; those differing identities
remain recorded, rather than claiming identical execution identity. An earlier
comparison record omitted nested model identity fields because its temporary
normalization shared an object reference. The corrected comparator copies those
fields before normalization; the earlier record is retained and superseded.

The published Q4K observation uses batched GPU decoding, whereas both F16 runs
use self-speculation with command batching disabled. A subsequent diagnostic
changes only resolved `session.speculation.mode` to `none` on the corrected
artifact. It preserves the same ordered cases and frozen browser source. Its
completed run confirms `single_token` decoding for all eight cases, with clean
teardown. All eight token sequences and output strings remain exactly identical
to corrected attempt 02. The hypothesis that disabling self-speculation removes
these failures is therefore refuted; no workaround is promoted.

`no-speculation-browser.tar.gz` retains the run, all 591 captured source files,
configuration, log, and executable comparator. The comparator rehashes both runs'
captured sources, checks identical model artifacts and ordered inputs, and
requires the actual decode modes to differ as intended. It retains the differing
server addresses and session identities. Other exposed runtime-profile fields
match; this is not a claim that the complete resolved sessions are identical.
Reproduce after extracting this archive and `corrected-browser.tar.gz` into the
same directory:

```sh
node compare-f16-speculation.js . rechecked-speculation.json
```

The first observational run completes all eight cases with identical tokens to
the uninstrumented single-token control and clean teardown. Its sixteen sampled
prefill embedding values match the source for each case at the trace's displayed
four-decimal precision; this is not full-tensor equivalence. It captures zero
requested `logits_final` probes: that stage is not the normal GPU output stage,
and GPU output finalization also omits configured probe forwarding. Therefore
this run cannot compare the required logits boundary. It is retained in
`embedding-probe-browser.tar.gz`, including its configuration and browser code.

The isolated source candidate forwards configured probes through GPU logits
finalization and releases the output on an observation failure. Regression
tests reproduce both the missing handoff and failure-path leak before repair;
the repaired orchestration and 170 inference test files pass. The next physical
run requests the `logits` stage and uses the same frozen browser closure with
only `logits/output-transform.js` changed. The complete candidate workflow passes,
including 790 test files. `logits-probe-repair-checks.tar.gz` retains the failing
handoff/cleanup regressions, repaired source and tests, inference suite, package
inventory, initial unpacked-budget failure, and complete passing workflow log.
The physical run completes all eight cases with clean teardown and captures one
logits probe for every generated token. Its outputs remain identical to the
uninstrumented single-token control. `logits-probe-browser.tar.gz` retains the
complete run and captured source. This repairs diagnostic visibility, not the
answer defect.
`probe-package-audit.json` accounts for the exact 192-byte unpacked increase in
one shipped file; file count and packed-size limits remain unchanged.

The completed CPU source replay in `source-step-reference.tar.gz` preserves all
eight original source token sequences while recording each generation step's
top eight scores, stop-token/newline scores, and full score-vector digest.
These are Transformers `output_scores` observations after generation processors,
not a claim to capture every raw intermediate activation. Source identity and
all input tokens are checked before execution. On the museum case, after the
eight shared answer tokens, the source ranks token 248046 at 22.861713 and
newline 198 at 21.697491. In the reused browser session, those scores are
19.9951 and 20.2742 respectively: the ranking has inverted before token
selection. The browser then repeats to the 256-token limit.

## Session-order control

The same museum question, run first in a fresh browser/model session, emits nine
tokens and stops at token 248046. Its stop-token score is 22.8625 and newline
score is 21.6960, close to the CPU source observation. It matches the source's
token sequence through the chat-end token. The runtime stops at chat end while
the source control continues its formatting suffix, so this is not full raw
token-sequence identity. The answer still omits required information and is not
task-qualified.

`session-reuse-browser.tar.gz` retains the fresh control and executable pair
comparator. The comparator checks completed runs, clean teardown, identical
model inventories, browser/device identity, load/generation/observation policy,
the exact question and passages, and all 591 captured source files in each run.
It verifies one logits observation per generated token. The differing condition
is the museum question's position: third after ticket and sensor, versus first
in a fresh model session. The full eight-case corpus remains the regression
corpus; the one-case arm is a reduced diagnostic, not a replacement evaluation.

After extracting `logits-probe-browser.tar.gz`, `session-reuse-browser.tar.gz`,
and `source-step-reference.tar.gz` together, reproduce with:

```sh
node compare-f16-session-reuse.js . rechecked-session-reuse.json
```

Fresh extraction and comparison pass. Browser values are displayed to four
decimal places; source step scores are generation-processor output. Concurrent
local activity prevents performance claims. This establishes session-order
dependence of the failure, not the identity of the faulty buffer or state owner.
The next paired probes inspect layer-zero projections, linear-attention output,
and completed-layer output under the two job histories. No numerical kernel or
state-reset repair is included yet.

## Layer-zero session-reuse boundary

`layer-reuse-browser.tar.gz` retains paired physical runs with the museum case
first versus third after ticket and sensor. Both completed with clean teardown,
the same corrected model inventory, generation controls, observation settings,
and 591 byte-verified browser source files. No numerical kernel or reset change
was applied. The answer again stops at nine tokens fresh and repeats to 256
tokens after the preceding jobs.

Across prefill and eight decode steps with identical input tokens, the displayed
embedding and layer-zero QKV samples and statistics match at all nine checkpoints.
The linear-core output statistics differ at all nine checkpoints; completed-layer
and logits observations also differ. The first core checkpoint has equal sampled
dimensions 0–15 but different whole-row statistics, so those samples cannot be
used to claim complete tensor equality. This localizes the earliest observed
divergence to the core boundary, not yet to a particular state buffer or kernel.

The comparison rehashes all captured runtime files and checks artifact, prompt,
generation, device, and source identity before comparing observations. Extract
the archive into a fresh directory, then run:

```sh
node compare-f16-layer-reuse.js . f16-layer-fresh-01 f16-layer-reused-01 replayed.json
```

Fresh extraction and comparison passed. These are diagnostic development cases,
not untouched holdout evidence or independent-machine participation.

`core-input-reuse-browser.tar.gz` repeats the pair with additional existing Z,
A, and B projection probes. Both runs completed with clean teardown and identical
source/model controls. All nine displayed checkpoints match for embeddings and
each of QKV/Z/A/B; all nine core-output checkpoints still differ. The same
comparator passed after fresh extraction using `f16-inputs-fresh-01` and
`f16-inputs-reused-01`. This narrows the next observation to the core's state and
parameter buffers, without claiming all projected tensor elements are equal.

## Reconciled candidate failures

The isolated candidate merged concurrent upstream `810b91fd` as `6877e02e`,
preserving its GELU, conversion, qualification, and tooling changes. These changes
do not enter the frozen browser experiments above. The merged check first failed
the measured package ceiling, then the stale generated execution-routing audit.
`reconciled-package-audit.json` accounts for all 34 changed shipped files by
SHA-256, including 24 same-size updates. Ten files contribute 3,571 additional
unpacked bytes. The package remains 1,780 files with unchanged dependencies;
ceilings now equal the measured 2,108,788 packed and 10,873,696 unpacked bytes.

Regenerating the routing audit adds 17 GELU digest mismatches against retained
manifests, taking surfaced integrity failures from 32 to 49, and removes none.
It does not rewrite old manifests, authorize new kernels for old Capsules, or
turn mismatches into verified model claims.

The subsequent complete check failed **5 of 793 test files**: Capsule naming
migration, Gemma 4 Q4-head conversion, Gemma 4 INT4-PLE variant identity, bundle
CLI, and Program Bundle exporter. Each failure reaches a changed GELU digest
or its derived conversion identity. `reconciled-checks.tar.gz` preserves all
three failed check logs and both npm package inventories. The merged candidate
is not green; the earlier passing 790-file workflow applies only to the earlier
probe-repair candidate. No failure was skipped or converted into a pass.

## Pre-core state observation

`state-reuse-browser.tar.gz` retains the completed pair with six opt-in
`linear_state_*` probes. Four parameter-buffer observations match throughout:
convolution weights, dt bias, A log, and normalization weights. Fresh museum
prefill starts with 24,576 zero convolution-state elements and 262,144 zero
recurrent-state elements. Third-job museum prefill starts with nonzero state:
convolution max absolute value 23.0348 and recurrent max absolute value 8.5972.
The recurrent-state observations differ through all nine identical-input
checkpoints. The answer remains nine tokens fresh versus 256 reused.

The new adapter only observes existing GPU buffers before core dispatch. It
uses declared logical element counts, not pooled buffer capacity; it does not
reset, upload, or modify state. Ordinary execution and unrelated probes do not
enter it. Configure these probes with an explicit trace category, as the retained
configs do. The unit regression checks buffer identity, geometry, forwarding,
non-mutation, opt-in dispatch, and failure propagation. Physical captures verify
the readback path, not a state-management repair.

Fresh extraction and `compare-f16-state-reuse.js` replay passed. The comparator
checks local token indices as well as source/input/config identity. Out-of-range
sample coordinates remain explicit strings, never numeric evidence or silent
NaNs. Raw-handle generation and explicit reset semantics still require a
lifecycle control before assigning the repair to runtime or caller. This result
does not establish failure of the signed Capsule execution path.

Upstream `b063f539` was reconciled as `90357cdd`; its retained-identity fixture
repairs pass all five previously failing tests without changing old manifests.
With the observation adapter, the complete `npm run check:green` workflow passes
795 test files. The focused inference suite passes 172 files. The complete logs,
focused regressions, exact observation source, and package inventory are retained
in `state-probe-checks.tar.gz`. `state-probe-package-audit.json` accounts for the
789-byte addition in one existing shipped module, with the file closure unchanged.
These passing checks supersede the earlier merged-check failure for this
candidate only; the failed receipts remain intact.

## Complete-prompt state repair

`reset-lifecycle-control.tar.gz` preserves the next physical control: the same
three jobs in the same order, same loaded model, same 591 runtime source files,
and unchanged prompts/settings, with explicit public `resetGenerationState()`
before each prompt. Museum generation changes from 256 tokens to nine. Every
recorded museum checkpoint, including initial state and all observed logits,
matches the fresh run. The first two answers retain their original token counts.
Both control comparators pass after fresh extraction with
`state-reuse-browser.tar.gz`. This tests raw-handle lifecycle, not a signed
Capsule or semantic qualification.

The owning normal-generation entry points, `_generateTokensInternal()` and
`generateTokenIds()`, reset decode bookkeeping but previously retained KV,
sequence position, and recurrent context from the preceding full prompt.
They now call the existing `_resetReplayPrefillRuntimeState()` after validation
and before beginning generation. Loaded model weights remain resident. Explicit
prefix-KV and incremental-decode methods are unchanged. Busy and unloaded calls
still reject before resetting any owned state.

The new regression runs the real generation entry points and reset owner,
stopping before GPU work. With the original source it fails because zero KV
clears occurred instead of one; with the repair both entry points clear KV,
reset sequence position, and empty prior linear runtime state. The initial
test-harness import-name error is retained separately from that valid regression
failure and is not evidence of the runtime defect.

`prompt-reset-browser.tar.gz` repeats the three-job physical run with the original
caller and **no explicit caller reset**. Its frozen runtime changes only the two
reset call sites; the other 590 served source files are byte-identical. Each
comparator verifies the exact source transformation, not merely an allowlisted
filename. Museum generation stops at nine tokens, and every recorded checkpoint
matches the fresh control. Fresh extraction and both comparators pass using
`state-reuse-browser.tar.gz` as the preserved baseline.

The same repair in the reconciled current source passes 173 inference test files
and the complete 796-file `check:green` workflow. `prompt-reset-checks.tar.gz`
retains both failure logs, passing logs, exact current source, regression, and
package inventory. `prompt-reset-package-audit.json` accounts for 86 additional
source bytes in one existing file. No kernel arithmetic, prompts, model bytes,
acceptance thresholds, or continuation APIs changed.

This closes the measured full-prompt state-contamination defect, not assistant
quality. The causal browser proof deliberately uses the earlier frozen runtime;
the actual newly packed candidate must also complete its separate eight-question
screen and eventual signed-Capsule/held-out qualification. The nine-token museum
answer still omits required facts and is not a useful-answer success.

## Claim boundary

This establishes a conversion-code defect and its focused repair. It does not
establish that truncation caused the Q4K browser's invented carton weight, that
the new F16 model answers correctly, or that the assistant is qualified. Task
quality still requires physical execution, unchanged acceptance, an untouched
holdout, and independent semantic review. The old F16 artifact is retained as an
adverse observation and is not admitted as a source-rounded precision control.

The archive omits weights. It contains source snapshots, tests, configuration,
sampled observations, and completed check logs. Unrelated dirty model-session
and structured-generation work in Doppler is preserved and not part of this
repair.

*Last updated: September 2026*
