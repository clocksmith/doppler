# Exact-byte diagnosis, component isolation, and opening

Both compact candidates remain rejected. No new model was signed, published, or
substituted into the accepted starter. This increment diagnoses the retained
failures; it does not start another model search. The preceding evidence exists
at commit `5825e4f2`, after `7ac72347`; the earlier 866-file validation is retained
in [that increment](../document-search-compact-2026-09-26/README.md).

## Qwen: quantization approximation accounts for the failed gate

The independent CPU oracle uses `gguf==0.19.0` to decode the actual candidate's
Q4_K bytes, then loads those decoded tensors into Hugging Face Qwen3 with float32
eager execution. It verifies the candidate manifest, every shard size and SHA-256,
and the original source files. It does not requantize the checkpoint. Token IDs,
attention masks, unprefixed inputs, last-token pooling and L2 normalization match
the frozen source contract. All 18 inputs are compared against both the original
reference and the previously installed browser execution.

| Comparison | Maximum component error over 18 embeddings |
| --- | ---: |
| Independent exact-quantized weights vs original reference | 0.05318811 |
| Doppler vs independent exact-quantized reference | 0.00011544 |
| Original frozen acceptance limit | 0.02000000 |

See `q4-reference.json.gz` and `q4-sensitivity.json.gz`. The independent decoder
is the external [GGUF package](https://pypi.org/project/gguf/0.19.0/), not a Doppler
codec. Seven representative packed weight blocks also match Doppler's diagnostic
reference decoder exactly; the bytes and independent expected values are frozen
in [the regression fixture](../../tests/fixtures/qwen-q4k-independent-blocks.json).
This checks byte interpretation, not just final rankings.

Installed browser operator diagnostics compare complete first-layer projection
matrices and transformer blocks 0 and 27 for the first frozen input. The embedding
slice matches exactly; first input-norm error is 3.58e-7; first Q/K/V projection
errors are at most 9.54e-6. Quantized block-output maximum errors are 0.000653 and
0.099304, versus 0.001035 and 0.160645 in the passing F16 control. Raw block
magnitudes differ greatly from normalized embeddings; these are diagnostic
measurements, not newly invented pass thresholds. See
[boundary-comparison.json](boundary-comparison.json).

The first material approximation appears at the first quantized projections:
independent Q-projection vs original has maximum error 0.26907, while Doppler vs
that same quantized projection differs by 8.58e-6. No unexpected equivalent-
computation discrepancy has been localized in the captured boundaries. This
supports quantization sensitivity as the explanation for the failed embedding
gate; it does not prove every operator on every input correct.

Restoring each of seven tensor groups independently to source-derived F16 yields:

| Restored group, all layers | Maximum embedding error | Added weight bytes |
| --- | ---: | ---: |
| Attention Q | 0.051276 | 84,410,368 |
| Attention K | 0.059616 | 42,205,184 |
| Attention V | 0.053251 | 42,205,184 |
| Attention output | 0.057729 | 84,410,368 |
| FFN gate | 0.052194 | 126,615,552 |
| FFN up | 0.058098 | 126,615,552 |
| FFN down | 0.031341 | 126,615,552 |

The down-projection group is the best measured precision-restoration direction,
but none passes 0.02. These are CPU sensitivity experiments, not newly converted
or qualified releases. No tolerance or reference was relaxed. The 300 MB target
remains provisional and was not used as an all-or-nothing numerical gate.

## MiniLM: retrieve and rerank independently

The matrix reuses `createDocumentSearch()` and rebuilds each embedding index. It
replays complete captured vectors and scores, without another search engine.
Incumbent values come from the exact installed signed models on Node/WebGPU;
MiniLM values come from the retained independent CPU source harness. These mixed
execution sources isolate application components, not cross-backend latency or
MiniLM Run support. No CPU-only incumbent claim is made.

Using the explicitly corrected development annotation v2:

| Embedding | Reranker | Relevant candidate found | Correct first result |
| --- | --- | ---: | ---: |
| Incumbent | Incumbent | 18/18 | 18/18 |
| MiniLM | Incumbent | 18/18 | 18/18 |
| Incumbent | MiniLM | 18/18 | 17/18 |
| MiniLM | MiniLM | 18/18 | 17/18 |

MiniLM's reranker chooses the embedding-change ticket for the `version` query,
instead of the release ledger. The same error remains with an identical exhaustive
passage list supplied to both rerankers; that separate diagnostic is explicitly
not retrieval success. Identifier queries and both long-document cases rank
correctly on this development set. The selected long passages contain the
requested cabinet identifiers and recovery phrases, not merely the right parent
file. Original v1 observations and rejection remain untouched.

See [component-matrix.json](component-matrix.json), including per-category counts,
raw ranking, retrieval recall, and the fixed-candidate diagnostic. On fresh
confirmation, MiniLM reranking ranks 8/8 answerable queries correctly and the
incumbent ranks 7/8, with either retriever. The incumbent confuses BX-210 with
BX-120. Thus the localized development failure is not a general claim that
MiniLM's reranker is always worse; neither comparison establishes a supported
mixed-model release.

## Abstention: frozen and evaluated, not qualified

`assessRelevance()` in the experimental evaluator distinguishes `match`,
`abstain`, and `unassessed`, binds a qualified policy to model/scoring/search/
preprocessing identities, and uses “No sufficiently relevant result found.”
Missing or unqualified policy means `unassessed`. This code remains outside the
shipped starter. Ranking is reported independently from acceptance.

Development selects a model-specific highest-score threshold that maximizes
useful acceptance subject to zero false acceptance. It requires at least 90%
useful acceptance; always abstaining fails. No first/second score gap or sigmoid
rescaling is introduced. Thresholds and confirmation-corpus hashes were frozen
before confirmation scoring in [abstention-frozen.json](abstention-frozen.json).

Calibration itself fails: incumbent threshold 21.378986 accepts only 4/18 useful
answers; MiniLM threshold 7.924092 accepts only 3/18. Both avoid the three
calibration false positives. The high-scoring missing-cabinet near miss drives
this tradeoff; ranking alone does not imply calibrated relevance.

Fresh confirmation contains eight answerable and eight no-answer queries,
near-miss identifiers, topical non-answering passages, duplicate relevant
passages, and relevant text near the beginning, middle, and end of long documents.
The same frozen policy is evaluated with 21-document and 10-document corpora.
At both sizes, the incumbent threshold accepts 0/8 useful answers and 0/8 false
answers; MiniLM accepts 1/8 useful answers and 1/8 false answers. All four pair
policies are rejected. See [abstention-confirmation.json](abstention-confirmation.json).

The corpus and labels are author-created, not independent external evaluation.
Future policy changes require fresh confirmation. These failures do not
retroactively invalidate installation, lifecycle, or previous ranking evidence.
The accepted starter still ranks results and has no qualified abstention policy.

## Unchanged accepted pair: less allocation during verification

The retained opening profile showed legacy secondary-digest compression and
allocation dominating CPU time. `src/storage/blake3.js` now reuses block words,
compression output and the chaining value within a chunk. Only the final block's
input state is retained for root/tree output. Scratch remains owned by the chunk;
no shared global state, skipped check, trusted persisted-file flag, or changed
identity was introduced. Historical digests remain byte-identical. The existing
nonstandard legacy digest is preserved; this is not a standards correction.

The new package was packed, installed through the package smoke, then vendored
into a new application archive and extracted into a separate consumer with frozen
`npm ci`. No checkout runtime sources were copied into that consumer afterward.
Only one shipped file changes: 403 additional unpacked bytes, no new files or
dependencies. The package budget increase matches that audited delta exactly.
See [package-audit.json](package-audit.json) and [package-receipt.json](package-receipt.json).

Sequential ABBA measurements use fresh processes, warm OS filesystem cache,
persistent disk and the same accepted embedding/reranking pair:

| Arm | Opening times | Median |
| --- | --- | ---: |
| Previous optimized loader | 17.956 s, 18.180 s | 18.068 s |
| Chunk scratch reuse | 12.393 s, 12.509 s | 12.451 s |

The scoped median reduction is 31.1%. Repeated complete searches, unchanged-index
reuse, output comparison and closure pass in every run. Candidate process peak
RSS is 1.842–1.848 GB; requested GPU allocation peak remains 3.987 GB and is not
physical residency. Two runs per arm establish this local diagnostic, not a
population-wide performance claim. See [opening-summary.json](opening-summary.json).

The installed candidate also passes cancellation after GPU submission, superseded
queries, interrupted saves, explicit closure, corruption repair and device-loss
recovery (`lifecycle.json.gz`). A separate fresh-process restart with IPv4/IPv6
socket creation denied by seccomp passes the same retained search results
(`offline.json.gz`). Kernel rejection is recorded as `EPERM`; this is not merely
a mocked fetch failure. Neither probe is a machine reboot.

The old accepted application archive remains immutable. This is a separately
installed loading candidate over retained signed releases, not renewed eligibility
or fresh-acquisition acceptance. Browser opening, reboot, smaller-memory hardware,
minimum memory and independent adoption remain unestablished.

## Reproduction and evidence boundaries

Policies next to this report provide explicit paths and inputs for the independent
CPU reference, browser operator capture, source component captures, matrix,
calibration/confirmation, package build, opening, lifecycle and offline probes.
Local model/package paths must exist. Outputs use exclusive creation; choose new
output paths for reproduction rather than overwriting observations.

The full large browser receipts remain in `/dev/shm/doppler-diagnosis-20260926`.
Their hashes, installed identities, selected operator metadata and final vectors
are retained in `gpu-boundary-receipts.json`; selected complete GPU tensors and
CPU tensors are retained as NPZ files. These are extracts, not full receipt copies.
The first oracle attempt failed on an uninitialized non-persistent RoPE buffer;
that setup failure is retained. Recreating that buffer through the upstream model
constructor corrected the oracle. A preliminary trace-probe attempt captured no
recorded intermediate values; the subsequent operator-diagnostic capture does.

Component: `doppler.runtime-source.storage`, `doppler.repository-tooling`,
`doppler.tests`, `doppler.docs`.
Intent: preserved. Boundary effects: none; search-policy experiments stay outside
Run and the accepted application, and Rig → Capsule → Run remains intact.
Acceptance evidence: retained numerical comparisons, component/confirmation
reports, installed package checks, opening probes, and the validation receipt.

`npm run check:green` passed all required gates and **868 test files**. Focused
independent-decoder and abstention regressions pass, including failed calibration,
all-abstain rejection, score validation, and incompatible-policy binding. Final
document/dependency checks, Python syntax checks and whitespace checks pass. See
[validation.json](validation.json) and the retained complete check log. The final
evaluator reproduces every development ranking decision and frozen threshold in
`component-matrix-reproduced.json`. These passes do not promote the rejected
models or abstention policies.
