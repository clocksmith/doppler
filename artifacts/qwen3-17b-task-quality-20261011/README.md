# Qwen3-1.7B source candidate: rejected

The pinned original source model passes **4/6 short fresh tasks** and **0/4
supplementary long-output/document tasks**. All ten terminate naturally. No
protected acceptance requests were replayed. No conversion, GPU execution,
resident integration, model promotion, runtime change, or release-byte change
followed this rejection.

## Frozen requirements and result

The short suite was frozen before its run; the supplementary suite was frozen
before its own run. Both require exact expected text, ignoring whitespace, plus
the declared EOS. They retain greedy generation, full-history repetition penalty
1.1, 4096 output tokens and 8192 context tokens. The official source template has
thinking explicitly disabled, as recorded before execution. No system instruction
or output rewriting was added.

| Task | Result |
| --- | --- |
| Ascending 84–99 | Pass |
| Descending 62–48 | Pass |
| Duplicate cherry/elm values | Pass |
| Literal amount 81.25 | Pass |
| Negated permission | Fails exact format: `No.` rather than `No` |
| 26 plus 7 | Correct value 33, fails requested integer-only format |
| All 220 integers, 211–430 | Fails: ellipses, values beyond the range, ends at 440000 |
| All 215 integers, 560–346 | Fails: omits the interior with an ellipsis |
| Conditional payment deadline | Fails meaning: `Yes [D7]`, expected `No [D7]` |
| Approved amount with passage citation | Correct amount/citation, fails exact format |

Thus short formatting disagreements are not the only blocker. The longer
control exposes missing work and an incorrect reading of a condition. The prior
general formatting-instruction candidate was not evaluated here: the declared
prerequisite that the baseline preserve long/document meaning did not pass.

## Provenance and numerical scope

Source: `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
Both original SafeTensors files match the official immutable-revision LFS
SHA256 values; total source weight bytes are 4,063,515,592. The receipt identifies
every downloaded config/tokenizer/weight file. Source metadata and the model
index are retained; weights remain outside Git.

The independent Transformers Qwen3 implementation loads original source weights
as exact F32 representations, uses eager source arithmetic and stores attention
K/V in F16. This is not a native-BF16 default-generation claim and does not
establish Doppler numerical equivalence. The tokenizer IDs, complete generated
tokens, raw/penalty-adjusted first-step logits, template rendering, dependency
versions, settings and stopping reasons remain in the reports.

The shared evaluator was extracted from the prior LFM diagnostic without runtime
changes. Its exact executed bytes and both exact entrypoint/evaluator digests
are retained. Historical LFM producers remain retained separately.
The focused same-input LFM regression preserves all six input/output token
sequences, first-step logits and verdicts exactly; its report is retained under
`../lfm25-task-quality-20261011/shared-evaluator-check.json.gz`.

## Capacity feasibility, not qualification

The official shapes are 28 dense layers, hidden size 2048, intermediate size
6144, 16 query heads, eight KV heads of dimension 128, vocabulary 151936, and
tied source embeddings. Existing Qwen3 kernels support these architectural
operations; a model-specific preparation recipe would still be required.

Static Q4_K matrix storage is 792,723,456 bytes; an F16 embedding is 622,329,856
bytes. A separately prepared Q4_K head is 175,030,272 bytes. F16 embeddings and
heads at both endpoints plus one full 8192-token cache would already exceed the
1,420,000,000-byte ceiling per contributor. With a Q4_K head and an 11/17 split,
weight-plus-one-cache lower bounds become 1,302,855,680 and 1,226,752,000 bytes,
before normalization, workspace, pools, staging and activations.

The existing resident API reserves each attempt's actual prompt-plus-output
allowance, not 8192 positions for every short request. Two short requests with
4096 output allowances therefore differ from two full 8192-position caches.
Neither this observation nor the static bounds establish physical acceptance;
the context, concurrency and per-device requirements remain unchanged. No cache
change was selected because source quality failed first.

## Reproduction

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python3 tools/source-model-task-quality.py \
  --source /var/tmp/doppler-qwen3-17b-source-20261011 \
  --suite artifacts/qwen3-17b-task-quality-20261011/fresh-suite.json \
  --out reports/local/qwen3-17b-quality-20261011/source.json --threads 4
```

Repeat with `long-document-suite.json` and a separate report destination for the
supplementary gate. Both commands exited successfully; their acceptance verdicts
are failures. Python compilation, pinned-file digest assertions, explicit task
counts/termination assertions and `git diff --check` passed.

Component: `doppler.repository-tooling`. Intent: preserved. Boundary effects:
none. These are retained source-model quality controls, not product acceptance.
