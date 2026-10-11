# Qwen3 source application-policy investigation

This investigation reuses the pinned original Qwen3-1.7B source model. It tests
specific application-policy hypotheses after the earlier source-quality
rejection. It changes no production model, generation policy, parser, numerical
fixture, release archive, or memory limit. Final execution disposition is in
`review.json`; the complete generated sequences remain in compressed reports.

## Repetition penalty at the first omitted range

The retained `extended-ascending` failure first emits an ellipsis at generated
step 12. Replaying the exact prefix preserves all thirteen selected tokens.
Before repetition adjustment, `...` already has logit 40.2576599, ahead of `2`
at 37.7889290. Full-history penalty 1.1 lowers the latter to 34.3535728 without
changing the winner. This boundary does not support blaming the omitted range
on repetition penalty. It does not establish that penalties never affect other
requests. See `ascending-ellipsis.json.gz` and the exact `probe-evaluator.py.gz`.

## General formatting instruction

`baseline-suite.json` and `formatting-suite.json` were frozen before their
paired execution. Both contain seven new tasks, including complete ranges with
more than 200 entries, negation, duplicate preservation and grounded citations.
The second arm adds the existing general formatting instruction verbatim from
Reploid's earlier instruction candidate. Both otherwise retain nonthinking
greedy generation, penalty 1.1, 4096 generated tokens and 8192 total positions.

Both arms pass **one of seven** tasks and naturally terminate every request.
Both omit the interior of the long lists with ellipses. The instruction changes
some formatting but does not satisfy the declared exact-output contract.
This instruction is therefore not promoted. Reports, logs, frozen inputs and
the exact shared evaluator are retained here.

## Recommended thinking policy

The pinned [official model README](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)
recommends thinking-mode temperature 0.6, top-p 0.95, top-k 20 and disabled
min-p; it warns against greedy generation in thinking mode. The candidate uses
that template and sampling policy, with no system instruction and no repetition
penalty. The local source generation configuration leaves the repetition
processor disabled (`None` in the installed Transformers configuration), which
is equivalent to multiplier 1.0. An initial setup assertion expecting literal
1.0 was corrected before model execution; this was a setup error, not model
quality evidence.

The existing limits remain **4096 generated tokens, 8192 total tokens and 16000
raw decoded UTF-16 code units**, matching the application's JavaScript character
count. The source README's larger suggested generation allowance is not adopted.
The frozen suite has seven new tasks under seeds 17 and 53. Every task must
produce exactly one natural `</think>` token, an exact final answer ignoring
whitespace, and a declared EOS within all limits. Full thinking and final-answer
tokens remain in the report; ignoring thinking for answer comparison cannot
remove it from token or character accounting. Budget exhaustion, missing or
duplicate closing markers, and wrong final answers reject the candidate.

The frozen suite SHA256 is
`0cf151811add3f9a61e931a1d3a24b1a3d6e64a878b2ee4ba617cf7f558b0b1f`.
The official README SHA256 is
`257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091`.
The generation-config identity is included in the frozen suite. A decisive
completed failure permits stopping remaining cases; those cases must be marked
unrun, never counted as passes or failures.

## Scope and delivery boundary

The original immutable source revision is
`Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
Both SafeTensors files match official LFS SHA256 values. Original weights are
represented exactly in F32, independent eager source equations execute on CPU,
and stored K/V rounds to F16. This is a declared source control, not the native
BF16 default or Doppler numerical qualification. The full source receipt,
tokenizer/template identities, package versions, input IDs and output IDs are
retained in every execution report. Original weights stay outside Git.

Doppler's resident interface currently delivers raw token/delta events without
reasoning/final channels. The distributed Reploid path has no qualified streaming
reasoning parser. Offline answer-boundary scoring here does not add one or
authorize a regex workaround. Passing source tasks alone would still leave
explicit model-owned streaming semantics, preparation, physical memory,
numerical and complete-application acceptance to qualify.

The narrower proposed integration keeps the existing resident result shape:
Doppler's prepared model-owned decoder emits final-answer deltas while tracking
the complete raw stream and rejecting malformed reasoning boundaries. Reploid
does not need a new reasoning UI or regex postprocessing. The current ordinary
decoder omits special tokens, so visible text length cannot supply that raw
accounting. `integration-assessment.json` retains the proposal and static
capacity calculations, with no implementation or physical acceptance claim.

An 11/17 split using Q4_K layer matrices, an F16 input embedding and a separate
Q4_K head has weight-plus-cache lower bounds of 1,302,855,680 and 1,226,752,000
bytes for 8192 combined reserved positions. Norms, workspace, pools and staging
remain excluded. An F16 head fails the capacity lower bound; actual Q4_K head
materialization and both execution phases must be checked. Existing actual
prompt-plus-output reservations and the 1,420,000,000-byte ceiling are unchanged.
Even complete original-source quality success could authorize only preparing a
candidate: deployed quantized weights would require their own independently
decoded-weight quality and numerical references before promotion.

Two additional read-only integration findings prevent treating this source
configuration as an already-supported drop-in runtime policy. The official
Qwen3 thinking template ends at the assistant header and lets the model emit
`<think>`; Doppler's existing Qwen thinking formatter appends `<think>\n` itself.
Those prompt token sequences differ. The source sampler also uses an evolving
PyTorch generator and multinomial draws. The resident's repeated scalar draw was
independently reproduced and repaired in `7e9cb892`; the separate
`../seeded-generation-sequence-20261011` evidence covers that fix. Its corrected
step-indexed scalar sequence still differs from PyTorch. Numeric seed equality
does not establish identical sampling semantics. Required independent fixed-token
numerical comparisons remain separate from stochastic quality tests.

`evaluator-boundary-check.py` provides six synthetic harness controls: single
boundary success, absent boundary rejection, duplicate boundary rejection,
UTF-16 character overflow, token-budget exhaustion without injected EOS, and
fixed-seed sampling. These check the diagnostic's accounting and stop semantics;
they do not independently qualify any model.

## Reproduction

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python3 tools/source-model-task-quality.py \
  --source /var/tmp/doppler-qwen3-17b-source-20261011 \
  --suite artifacts/qwen3-17b-application-policy-20261011/recommended-thinking-suite.json \
  --out reports/local/qwen3-17b-policy-20261011/recommended-thinking.json \
  --threads 4
```

The paired controls use the corresponding baseline/formatting suite and distinct
report paths. The retained-prefix probe additionally supplies `--probe-report`,
`--probe-case extended-ascending` and `--probe-step 12` against the original
supplementary report and its unchanged suite. Executed entrypoint/evaluator
snapshots are retained because the helper gained policy support during this
investigation; report digests bind each invocation to its actual code.

Component: `doppler.repository-tooling`. Intent: preserved. Boundary effects:
none. These controls cannot close integrated application or deployment gates.
