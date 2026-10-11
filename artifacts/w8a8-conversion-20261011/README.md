# Gemma 3 270M W8A8 conversion

The ordinary Safetensors converter emits 126 W8A8 matrices and 110 F16 tensors
from the pinned original `google/gemma-3-270m-it` source. Embeddings, tied head
and normalization storage retain the dense F16 control. The seven shard hashes
and all packed rows were verified. `result.json` binds the source, recipe,
manifest, shards and exact numeric policy. Large source/artifact bytes remain
outside Git; no model catalog or standard archive was replaced.

The first construction was repeated to declare the composed projection's F32
input/output explicitly in its execution graph. Both constructions have
identical shard bytes and tensor descriptors. The retained final manifest
identifies the graph used for subsequent independent/runtime checks.

Focused tests cover complete policy rejection, exact ties and neighboring F32
inputs, signs, nonfinite/subnormal rejection, partial-row padding, malformed
storage, preservation of dense embeddings, streamed row concatenation, and
upload failure cleanup. Packed outputs also match the separately implemented
BigInt reference. Existing Q4 and dense loader controls remain passing.

These are converter and synthetic loader results. Actual GPU model execution,
operator agreement, model approximation, application quality and installation
must be reported separately; conversion alone certifies none of them.
