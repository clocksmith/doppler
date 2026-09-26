# Compact search candidate screening

Neither tested candidate is ready to replace the accepted model pair. The
reference release and its immutable artifacts remain unchanged. This directory
retains negative results as well as the reusable evaluation path.

## Frozen evaluation

[corpus.json](corpus.json) contains 17 author-created documents and 21 queries:
18 answerable queries and three no-answer queries. It includes the original six
application examples, paraphrases, confusable exact identifiers, irrelevant
documents, and a long document with relevant material near its end.
[corpus-freeze.json](corpus-freeze.json) binds the bytes before candidate outputs
were observed. This small internal corpus is not independent adoption or a
general retrieval benchmark.

The explicit screening preprocessing contract uses 480-code-point windows with
80-code-point overlap. It retains original documents and records each passage's
document identity and start/end offsets. Passage identities include content and
the preprocessing contract. Inputs exceeding the source tokenizer's declared
limits fail instead of being truncated. This preprocessing is evaluation tooling;
it has not silently changed the shipped starter's indexing behavior.

Both source screening and installed execution call the existing application's
`createDocumentSearch()`. Source replay supplies captured vectors and pair scores;
it is not a second retrieval implementation. The installed probe imports the
installed `search.js`, validates the generated application/runtime asset inventory,
opens the existing model sessions, and indexes the screening corpus in memory.
It does not replace the user's persistent documents or index.

## Smaller-model experiment

The exact downloaded source inputs for
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
and [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a)
total **183,397,846 bytes**. Both upstream cards declare Apache-2.0. The receipt
pins revisions, file sizes and SHA-256 digests. This is source download size, not
the size of an implemented Capsule distribution.

The CPU source oracle uses float32 eager execution, mean/L2 embedding pooling,
and the cross-encoder's raw scalar logit. Its relevance threshold of zero was
declared before outputs were captured. No threshold was tuned on these results.

- Raw ranking top-1: **16/18**.
- Correct answer accepted at the fixed threshold: **13/18**.
- Identifier queries accepted correctly: **4/5**.
- Long-document queries accepted correctly: **2/2**.
- No-answer false positives: **1/3**.

The missing-cabinet query incorrectly accepts a passage about different cabinet
identifiers. Several correct rankings fall below the fixed threshold. The pair
fails the frozen gate; it was not prepared as a new supported Rig model family,
signed, substituted into the starter, or published. See
[minilm-screen-final.json](minilm-screen-final.json) and
[source-screen-receipt.json](source-screen-receipt.json). The earlier
`minilm-screen.json` predates the additive raw-ranking metric; its decisions are
unchanged. `minilm-source.json.gz` retains complete source vectors, token IDs,
pair scores, source identities and source timings. These are not Run timings or
WebGPU evidence.

## Existing-model quantization experiment

The existing Qwen Q4K recipe was used to prepare a **new** embedding identity from
pinned source bytes, with standard SHA-256 shard hashing. It emitted nine shards;
every size and digest was independently checked. Model-directory bytes total
**563,967,820**. Its F16 embedding table alone occupies **310,618,112 bytes**, already
above the provisional combined 300 MB target before any reranker is included.

The candidate failed the existing browser source-reference gate:

- All 18 token sequences match exactly; repeated outputs are stable and finite.
- Every output vector exceeds the frozen maximum-component-error limit of 0.02.
- Maximum observed component error is **0.0532196**.
- The F16 control passes all **36** token/vector checks using the same installed
  runtime and execution environment. The source revisions differ in README bytes;
  their weight/config/tokenizer digests and frozen reference vectors are identical.

This narrows the discrepancy to the quantized path. It does not yet distinguish
expected quantization loss from an implementation defect. No kernel, source
reference, or tolerance was changed to produce a pass. See
[qwen-receipt.json](qwen-receipt.json), `qwen-qualification.json.gz`, and
`f16-control.json.gz`.

An additional Node application diagnostic loads this unsigned Q4 embedding
candidate with the existing signed reranker, using the installed runtime and
search implementation. It ranks **17/18** answerable queries correctly and both
long-document queries correctly. That application result does not overrule the
failed numerical gate or establish signed execution of the new pair. See
[qwen-screen.json](qwen-screen.json). No new Capsule was signed or promoted.

## Incumbent and annotation limits

The retained signed pair also ranks **17/18** answerable queries correctly, with
all identifier and long-document queries correctly ranked. Its existing search
contract always returns ranked candidates: it has no qualified no-answer decision.
Consequently all three no-answer cases fail that new requirement in this probe.
The practically unbounded incumbent score threshold records that existing behavior;
it is not compared to MiniLM's zero threshold as a calibrated confidence score.

The `upgrade` query exposed an annotation ambiguity after the corpus was frozen:
both `recipe` and `ticket-4821` state that vectors need rebuilding when the
embedding space changes, but the expected relevant set names only `recipe`.
Both models choose `ticket-4821`. Preserve the original result; do not treat this
one label mismatch as proof of a model defect. [corpus-v2.json](corpus-v2.json)
corrects this relevant set; its separate freeze records that the correction was
informed by observed results. All reports and selection decisions here retain
version 1; version 2 is not unseen evaluation data. Even discounting this row,
the other failures remain.

Both resident models execute on the current Linux/AMD reference machine. These
quality diagnostics overlap repository testing and conversion work; their opening
and inference timings are not controlled performance comparisons. Source CPU
timings, browser qualification timings, and installed Node timings are distinct.
No smaller-memory machine, reboot, ordinary-browser configuration, minimum memory,
independent adopter, or release publication is established here.

## Reproduction and handoff

The corpus tool accepts a JSON config with `mode: "prepare"`, `corpusPath`,
`freezePath`, and `outputPath`; it refuses a changed corpus hash. Capture pinned
sources with `tools/capture-compact-search-source.py` and the retained source
policy. Then use `mode: "source-replay"` with `capturePath` and `preparedPath`.
The compressed prepared inputs are retained as `prepared.json.gz`.

The installed probe accepts the retained `incumbent-final-policy.json` or
`qwen-screen-policy.json`; local installation/cache paths must exist. New outputs
use exclusive creation so observations cannot be silently overwritten. Browser
qualification uses the existing `tools/qualify-embedding-browser.js` with the
retained browser policies. The Qwen conversion config and original conversion log
are retained; artifact bytes remain local under `/dev/shm/doppler-compact-20260926`.

Component: `doppler.repository-tooling`, `doppler.tests`, `doppler.docs`.
Intent: preserved. Boundary effects: none. Acceptance evidence: frozen corpus,
source capture, installed search reports, browser source/control qualifications,
and `tests/tooling/document-search-evaluation.test.js`. The regression verifies
complete Unicode passage coverage, stable identities after title-only edits,
invalid overlap rejection, reuse of application search, and no-answer failure.

`npm run check:green` passed all required gates and **866 test files**. The complete
log is retained in `check-green.txt.gz`, with hashes in
[validation.json](validation.json). Generated dependencies, goal contracts and
whitespace checks also pass after the final tooling/documentation edits. Repository
green status does not change either candidate's failed acceptance result.

Next work should address the measured issues: an independently calibrated
no-answer contract, using the corrected annotation for further candidate selection,
and a compact pair that passes both numerical and application quality gates.
Smaller hardware remains unavailable; its acceptance stays pending.
