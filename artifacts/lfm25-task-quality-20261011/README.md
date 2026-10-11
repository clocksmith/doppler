# Existing LFM2.5 artifact: rejected fresh-task screen

The existing local `lfm2-5-1-2b-instruct-q4k-ehf16-af32` artifact passes **3/6**
fresh tasks through independent Transformers Lfm2 equations. All six terminate
on their declared EOS. This candidate is rejected before GPU or resident
integration. No production weights, manifests, package bytes, arithmetic,
acceptance fixtures, tolerance, or allocation ceilings changed.

The suite was frozen before execution; its SHA256 is
`a85d4dd9e78cf3bcf927b32c9d31997d3941f1ccfb1bfcc9c44184a800e4779a`.
It preserves the existing greedy, full-history repetition-penalty 1.1 policy and
4096-token allowance. Exact text, ignoring whitespace, and natural termination
are required for every task.

| Fresh task | Result |
| --- | --- |
| Ascending 46 through 63 | Fails: starts at 47 |
| Descending 94 through 80 | Fails: starts at 98 and continues toward 0, skipping values |
| Copy repeated cedar/pine values | Pass |
| Preserve literal amount 62.40 | Pass |
| Respect negated permission | Pass |
| 18 plus 7 | Fails: answers 19 |

## Scope and provenance

The screen uses the already-local 851,352,832 bytes of stored weights.
Independent `gguf` Q4_K decoding, exact BF16/F16 conversion, strict source-model
state keys and tensor shapes, tied source embeddings/head, eager F32 equations,
and F16 attention-cache storage are explicit. Convolution state follows the
independent source implementation. The source config/tokenizer/template revision
is `LiquidAI/LFM2.5-1.2B-Instruct@0f604ada3f766f9f257460c4c9f0b5d6f69d431b`;
only metadata was downloaded. Deployed and source tokenizers produce identical
IDs for every rendered input.

This is an identified local-artifact screen, **not original pretrained model
qualification**. All thirteen legacy manifest BLAKE3 values differ from standard
BLAKE3. The result retains actual shard SHA256, standard BLAKE3, legacy fields,
exact manifest SHA256, source metadata digests and implementation versions. The
original source-weight lineage is unestablished; this screen does not repair or
promote the artifact. Its catalog status remains experimental/unpromoted.

Static split-at-eight layer storage is 294,998,784 and 287,914,496 bytes. Adding
the 268,435,456-byte tied embedding/head at each endpoint would give respective
lower bounds of 563,438,336 and 556,354,048 bytes, before cache, normalization,
workspace or transient allocations. This is feasibility arithmetic, not a
physical memory qualification or combined-capacity result.

## Repetition-policy observation

The observed repeat preserves every input and generated token from the first
run. At the first ascending step, raw `47` has logit 21.9073868 and raw `46` has
21.5277596. Penalizing `46`, which occurs in the prompt, lowers it to 19.5706902,
but does not change the already-wrong winner. Thus the proposed penalty cause
does not explain this omission.

The descending step is different: raw `94` wins at 19.3796272 over `98` at
18.1601486. The penalty lowers `94` to 17.6178436, making `98` win. This isolates
one application-policy effect without establishing that a neutral policy would
make the complete candidate acceptable. No policy change or protected-case
tuning was performed.

## Reproduction and retained evidence

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python3 tools/lfm2-deployed-task-quality.py \
  --model models/local/lfm2-5-1-2b-instruct-q4k-ehf16-af32 \
  --source /var/tmp/doppler-lfm25-source-metadata-20261011 \
  --suite artifacts/lfm25-task-quality-20261011/fresh-suite.json \
  --out reports/local/lfm25-quality-20261011/source-observed.json --threads 4
```

The initial and observed commands exited successfully; test acceptance fails.
`source-result.json.gz` and `observed-result.json.gz` retain all inputs, generated
tokens, termination and task verdicts. Their logs and the exact initial producer
are retained alongside the immutable source metadata receipts. The current tool
adds first-step logit observation only. Python compilation, exact input/output
comparison between runs, and `git diff --check` passed.

Component: `doppler.repository-tooling`. Intent: preserved. Boundary effects:
none. This is CPU source evidence, not physical execution, installed-package
acceptance, application release acceptance, or numerical-reference approval.
