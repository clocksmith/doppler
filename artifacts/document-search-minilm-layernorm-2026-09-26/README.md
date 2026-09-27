# MiniLM affine normalization prerequisite

This is a runtime prerequisite for BERT lowering, not MiniLM qualification.
The independent source reference and confirmation corpus remain frozen at the
[prior checkpoint](../document-search-minilm-embedding-2026-09-26/README.md).
The accepted search archives and model descriptors are unchanged.

## Inspected and changed

The configurable layer executor could express RMSNorm but not BERT's affine
LayerNorm after residual additions. Add an explicit LayerNorm operation to the
plan contract and preserve its affine selector from execution-v1 tuples. Reuse
existing LayerNorm shaders and parameter dtype validation; preserve residual
order through a separate addition. Load pre/post-FFN affine biases explicitly.
Missing parameters and implicit dtype changes fail before numeric execution.

A failed later step also exposed owned current-state buffers omitted from error
cleanup. Cleanup now releases every owned referenced buffer once and preserves
borrowed input. Recorded temporary buffers remain with their recorder until
cleanup. The same catch boundary covers final observation failures.

Move the cohesive layer/expert/router weight declarations into the existing
weight module and re-export them from the previous type entrypoint. This keeps
the reviewed declaration size bound without raising it or changing runtime imports.

## Execution and evidence

`source-identity.txt` pins the starting revision and changed executable files.
`focused-tests.log` retains physical immediate/recorded numerical checks on the
local AMD/Vulkan provider and the adjacent config/loader regressions. The
independent arithmetic fixture exercises residual-then-affine-normalization,
nonuniform scale and bias, and separate token rows. Failure cases cover a missing
later bias, partial affine upload failure, and an undeclared dtype change, with
pool activity and borrowed-buffer ownership assertions.

`checks.log` records source declarations, architecture, style, inference-boundary,
and unit-suite verification. The broad unit run caught a missing `RouterWeights`
type import after the declaration move; the import was repaired and the failing
consumer-declaration test rerun successfully. The original unit transcript and
the corrective rerun are retained separately. These checks are local diagnostics, not browser,
full-model, installed-application, or release qualification.

## Remaining work

Complete source-bound BERT input embeddings (learned position/type sum and affine
normalization), post-residual encoder lowering, and output extraction without an
invented final normalization or decode state. Then execute the retained MiniLM
reference and frozen installed mixed-pair search comparison. No MiniLM Capsule,
model parity, retrieval benefit, memory reduction, download reduction, or
independent adoption is established by this increment. Generation is unchanged.

Component: `doppler.runtime-source.inference.pipelines.text`,
`doppler.runtime-source.loader`, `doppler.runtime-source.config`.
Intent: preserved.
Acceptance evidence: focused tests and checks linked above.
Boundary effects: additive explicit layer-plan operation and affine bias loading;
no model-name dispatch, shader change, release archive change, or support promotion.
