# Isolated Qwen3-1.7B preparation

This artifact records diagnostic preparation, not an accepted application,
published model, signed release, or physical memory qualification. The original
source thinking-policy screen remains separately retained in
`../qwen3-17b-application-policy-20261011`.

The existing Rig converter prepared all 311 tensors into 24 SHA-256-verified
shards containing 1,590,331,392 bytes. The manifest digest is
`bc7ede46274d8c18ebb882bb30a87b7a31ccbf3e87afb691a63c476fa0cdcf38`.
The exact recipe, source state, converter output, manifest and shard verification
are retained here. Source geometry and RoPE theta remain those of pinned
Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

Layer matrices and the separate output head use Q4_K; the input embedding uses
F16. The head is 175,030,272 bytes and the embedding is 622,329,856 bytes.
Although the original checkpoint ties these endpoint weights, this deployed
precision profile has distinct tensors. The offline evaluator must load both
exactly and must never retie the quantized head to the F16 embedding. The prepared
tokenizer JSON is semantically identical to the original source tokenizer JSON.

`suite.json` freezes the exact manifest and the previously frozen original-source
tasks, settings, seeds, natural boundary/EOS requirements, 4,096 generated-token,
8,192 total-token and 16,000 raw UTF-16 character limits. Its SHA-256 is
`34ca02e794f8067c08bb19a0d1dbc84e39617557eeeff9432d004a459e338537`.
Reusing these cases assesses the conversion against the selected source policy;
it is not a new independent holdout or proof of GPU sampling parity.

The additive `--deployed` mode in `tools/source-model-task-quality.py` uses the
same source architecture, evaluator and tokenizer validation. Its offline loader
uses independent `gguf.dequantize` Q4_K decoding, verifies every shard and tensor,
and rejects missing, malformed or nonfinite data. Synthetic loader tests preserve
distinct embedding/head values and reject eight invalid-artifact controls.
`loader-validation.json` additionally records strict independent decoding of all
311 actual prepared tensors: all are finite, shapes match the original source
architecture, and head/embedding storage and values remain distinct. This load
check performs no generation or GPU work.

The exact manifest's 11/17-layer split plus 8,240 combined reserved F16 cache
positions has lower bounds of 1,305,114,112 and 1,230,246,400 bytes. The remaining
114,885,888 and 189,753,600 bytes must accommodate all excluded materialization,
workspaces, pools, alignment and in-flight resources. These calculations in
`capacity-lower-bound.json` cannot establish physical acceptance.
The 8,240-position example means two 24-token prompts with 4,096 output tokens
reserved for each. It does not mean two full 8,192-token reservations fit.
Longer concurrent prompts require their actual reservations and may correctly
reject under the unchanged limit; repeated 1,588-token controls are separate.

Original-source quality completed 14 of 14 cases before the first deployed
invocation. The independent deployed-weight run passed the two long ordered
lists and integer answer for seed 17, then failed the frozen permission-negation
case: its final answer was `No.` instead of the required `No`. The meaning was
correct, but the exact-output gate remains failed. The result used a natural
reasoning close and EOS, 279 generated tokens and 1,334 raw UTF-16 characters.
The fail-fast monitor interrupted the next case after one token; nine cases
were never started. `review.json` distinguishes completed, interrupted and unrun
cases. The complete raw report and interrupted process log remain compressed
here. No GPU execution or model promotion is recorded here.
Measured partition memory, exact deployed-weight quality,
numerical comparison and model-owned final-answer decoding remain separate gates.
The standard 0.6.27 archive, catalog and signed records are unchanged.

The next bounded configuration experiment is frozen in
`format-policy-holdout.json` (SHA-256
`f08c3e7a5d027371fa6adefcc212ebd54c16fbdc2c001561c29db3719fbf7143`).
It adds one general system instruction about following explicitly requested
final-answer formats; seven new cases use seeds 29 and 71. Model bytes, sampling,
precision, budgets and exact-answer comparison are unchanged. A complete fresh
pass must be followed by unchanged prior-case regression; no punctuation is
stripped or accepted retroactively. This configuration also failed its first
completed fresh case: `permission-approval`, seed 29, emitted `No.` instead of
`No`. It naturally closed reasoning and emitted EOS after 197 tokens, using 933
raw UTF-16 characters. The next case was interrupted after one opening token;
twelve cases were unstarted. The unchanged prior-case regression was not run
because the fresh gate failed. `format-policy-result.json.gz` and its process
log preserve the failure. At that first-policy checkpoint no further instruction
variant had been evaluated; the subsequently authorized literal-format screen is
reported separately below.

The instruction was refined while the first invocation was initializing. Its
interruption retained one opening marker token but no evaluated answer. A guard
then failed before rewriting the suite, and a shell continuation inadvertently
started another initialization; that was also stopped. The superseded suite,
partial report and both interrupted logs remain here. They are not candidate
quality results or evidence comparing two instructions. The revised suite was
written and hashed in a separate completed operation before the actual launch.

Component: `doppler.repository-tooling`. Intent: preserved. Boundary effects:
none; offline evaluation cannot change a running model's computation.

One attribution control tested the original rejected answer's punctuation
decision at generated-token index 277. It replayed every preceding token and
sampling draw exactly, captured the actual input to the deployed head, and
compared the Q4 and original higher-precision heads on those identical hidden
operands with cloned RNG state. Both selected `.` with probability 1 after the
unchanged sampling filters; both filtered out EOS. The deployed head reproduced
the actual logits exactly. This does not identify a head-specific selection
defect, nor exclude differences inherited from earlier computations.

`head-attribution-contract.json`, the complete report, log and exact executed
source snapshots retain that result. The helper's diagnostic top-ten listing
was subsequently capped for six-token synthetic test vocabularies; this does
not change the actual 151,936-token computation. Synthetic controls verify
same-head agreement, a deliberately changed head, exact RNG replay, mismatched
prefix rejection and observer cleanup. No model or quantization change follows
from this negative attribution result.

A subsequently authorized general literal-format instruction was frozen in
`literal-format-holdout.json` before execution, with eight fresh tasks and seeds
37 and 89. Five completed cases passed. The sixth emitted `$137.65[S8]` rather
than the frozen `$137.65 [S8]`, so the unchanged whitespace-token comparator
rejected it and the fail-fast monitor stopped execution. The prompt did not
explicitly require that separator. This is a fixture ambiguity, not evidence of
a wrong amount or citation. The rejected receipt remains unchanged; neither the
comparison nor the output was rewritten to pass. One following opening token is
retained as interrupted; nine cases were not started. Raw report, process log and
`literal-format-review.json` preserve the complete disposition.

The exposed fixture now has a prospective comparison correction in
`literal-format-comparison-v2.json` (SHA-256
`c60149d173e4be545d0f12ecdc7eb8a5ab218bdb20e490c951bffb02cd61f81e`).
It is not a new holdout. Prompts, expected strings, instruction, seeds, model and
generation settings are unchanged. Both cited-value cases explicitly compare
the exact value and exact bracketed citation, permitting only ASCII spaces or
tabs at their boundary after trimming outer whitespace. Internal whitespace,
wrong values or citations, extra punctuation and extra text still reject. Every
other case retains the previous comparator. Focused synthetic tests cover those
positive and negative boundaries and unchanged historical punctuation rejection.
The original failed receipt is not regraded. The exact v2 evaluator and entrypoint
are retained before the new invocation; its rejected result is reported below.

The frozen v2 suite's inherited top-level `scope` string still says fresh
holdout; its `comparisonRevision.scope` is the accurate classification. The
producer copies that stale top-level string into the report. Treat this run as
an exposed comparison-contract correction, as recorded here and in its review;
neither the suite nor completed execution bytes are changed to hide that
metadata inconsistency.

Two contingent prior-task regressions are frozen separately. Their first
versions preserve every original strict comparison; neither has been executed.
Independent review found the same undefined citation separator in four prior
case prompts. Prospective `literal-policy-original-comparison-v2.json` and
`literal-policy-first-format-comparison-v2.json` apply the same exact cited-value
class only to those cases. These are exposed comparison-contract corrections,
not fresh holdouts. All original prompts, expected strings, seeds, model bytes
and generation settings remain unchanged; the single selected literal-format
instruction is bound explicitly. They may execute only after the complete
sixteen-case v2 gate passes. Each result review must report the original strict
and corrected decisions on the same raw answers; corrected acceptance must not
be presented as passing a strict comparator that actually rejected it.

Contingent numerical observation contracts separately bind each regression's
first seed, all seven cases at steps 0, 1, 7 and 31, and supplemental steps 256,
512 and 1024 for both long-list cases. These are at most 68 full-vector
observations across both regressions; actual coverage must identify every skipped
position. The independent source uses the same decoded weights, pinned F32 eager
equations and stored F16 cache, with four CPU threads. It is not an emulator of
GPU reduction order or a replacement for the old Qwen3.5 reference. Binary vectors
remain outside Git with hashes in a compact index. The isolated observer's tests
verify unchanged model weights, logits, random state and sampled tokens, plus
identity binding, complete vector size, duplicate rejection and skipped coverage.
It has not yet been attached to the source evaluator or executed on this model.

`reasoning-runtime-manifest.json` is a separate inactive diagnostic manifest.
`reasoning-runtime-derivation.json` binds source manifest `bc7ede46…` to derived
manifest `a8647c3f…`: the only semantic addition is the explicit model-owned
reasoning-output contract. Existing ChatML, every numeric field, all shard
references and tokenizer bytes remain unchanged. Both marker IDs were verified
against the actual tokenizer. Any later independent-reference fixture must retain
the source identity and cite this derivation to its runtime identity. Preparing
this annotation does not activate a model or establish physical qualification.

The corrected sixteen-case invocation passed all eight seed-37 cases, including
both complete long lists, and the seed-89 positive-permission case. The next
negative-permission case emitted `No.` instead of the unchanged required `No`.
It naturally closed reasoning and emitted EOS after 218 generated tokens, using
307 total tokens and 1,036 raw UTF-16 characters. The candidate therefore fails.
The fail-fast monitor retained one opening token of the next case and left five
cases unstarted. Nine completed cases pass the corrected contract; eight pass
the original strict comparator. The latter additionally rejects the previously
documented citation separator. Both decisions are recorded on identical raw
answers in `literal-comparison-v2-review.json`; neither historical verdict is
rewritten. Complete result and interrupted process log are retained compressed.

The contingent prior regressions, numerical observer and inactive annotated
runtime manifest were prepared but not executed or activated. No observer was
attached to this run. The failed quality gate prevents advancement to physical
model qualification. No additional instruction or model variant follows from
this result, and the standard archive and old numerical fixtures remain unchanged.
