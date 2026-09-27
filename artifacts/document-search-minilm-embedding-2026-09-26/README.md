# MiniLM embedding-only preparation checkpoint

MiniLM remains **unqualified and unpublished**. This increment captures its
independent reference and source-truth ModelIR; it does not claim a working
Doppler MiniLM implementation or installed mixed-pair acceptance.

The selected source is `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The full source snapshot was copied
from temporary memory storage to `/var/tmp/doppler-minilm-embedding-20260926/source`.
The existing [capture harness](../../tools/capture-embedding-source-reference.py)
now admits its explicit mean-pooling/L2 contract alongside the unchanged Qwen
last-token reference. It validates source module order, dimensions, revision,
and the 256-token bound; it rejects oversized inputs instead of truncating.

`source-reference.json.gz` retains CPU float32 outputs for 12 fixed cases,
including empty/Unicode input, negation, identifiers, and a 256-token input.
Uncompressed SHA-256:
`0a3d5a6b8aec76564aa7d93fe472d648b18a8749d049b9bad045a8c731986cac`.
This uses the model's own PyTorch/Transformers reference, not Qwen vectors.

## Frozen application confirmation

Before further scores were captured, [confirmation-corpus.json](confirmation-corpus.json)
froze 60 documents and 72 queries: 60 answerable and 12 no-answer queries,
with paraphrases, identifier near misses, and relevant passages at the beginning,
middle, and end of long documents. Its [freeze receipt](confirmation-freeze.json)
pins SHA-256 `473ae5cef5c9d4630484c7b9fab8851de1982904895e13336e2acfb58eff9e64`.
Use the same versioned segmentation and incumbent reranker for both embeddings;
rebuild indexes per embedding identity. No-answer remains unassessed. This corpus
is author-created, not supplied by an independent developer, and has not yet been
scored through a Doppler MiniLM implementation.

## Concrete intake result

The first coordinator invocation rejected empty `stateSpaces`. BERT has no
persistent decoding state, so inventing a KV cache would misrepresent the model.
The ModelIR validator, declaration, and JSON Schema now accept explicit `[]`
for this field only; missing/null/malformed values still fail. Other required
node arrays remain nonempty. Focused regressions and validation of the actual
source ModelIR through the JSON Schema pass.

The second [onboarding result](rig-intake/onboarding-result.json) records a valid
source ModelIR with 104 tensor headers and 128 source facts, but blocks lowering.
It was evaluated against the retained **Qwen embedding-specific vocabulary**;
its incompatibility is not a universal inventory of Doppler's primitives.
The source contract requires learned position/type embeddings, affine embedding
LayerNorm, bidirectional attention, post-residual LayerNorm, nongated GELU FFN,
384-dimensional mean pooling, and no decode state. Those semantics require an
explicit compatible Rig lowering and physical qualification. Existing WordPiece,
LayerNorm, attention, and embedding primitives should be assessed for reuse.

Next: implement that bounded source contract, compare Doppler execution with
the retained reference, then qualify the installed MiniLM-embedding/incumbent-
reranker pair on the frozen corpus and lifecycle probe. No signed MiniLM Capsule,
distribution size, memory footprint, or WebGPU support is claimed here. The
MiniLM reranker remains a later candidate; the small historical screens do not
establish general inferiority. Qwen quantization and abstention remain separate
rejected research candidates.

The [package audit](package-audit.json) records the exact source/declaration/doc
byte increase with unchanged package file count. This newly packed runtime is
unpublished; the delivered loader maintenance archive remains byte-for-byte
unchanged at `c5be891b…`.

Component: `doppler.runtime-source.config`, `doppler.repository-tooling`.
Intent: preserved.
Acceptance evidence: ModelIR/source-truth regression tests, captured reference,
actual JSON Schema validation, and retained coordinator rejection.
Boundary effects: explicit stateless ModelIR representation; no Run algorithm,
Capsule support claim, public model promotion, or release-archive substitution.
