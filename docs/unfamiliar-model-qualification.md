# Unfamiliar encoder qualification

The ESM-2 8M checkpoint (`facebook/esm2_t6_8M_UR50D` at
`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`) exercises a complete six-layer
bidirectional encoder beyond the demonstrated Qwen decoder workloads. Its declared
outputs are all token embeddings and mean-pooled embeddings. The masked-language
model head and biological usefulness are outside this experiment.

Before GPU execution, three public sequences, every output element, source
PyTorch f32/eager execution, a 0.001 absolute-error limit, and a 30,000 ms recovery
budget were frozen. Initial preparation reused the existing ESM conversion route,
changing source identity, layer count, and head geometry. Source files were
downloaded at the exact revision and checked against upstream blob/LFS identities.

Full reference comparison found an implementation defect that partial probes had
missed: the source uses erf GELU, while the existing kernel uses tanh GELU. A GPU
readback on the first FFN input matched source tanh output within 0.000000311 but
differed from source erf output by 0.0004734. Changing the source reference to tanh
was diagnostic only; acceptance remained the original erf reference.

The repair adds explicit `GELU_ERF` specialization to the shared shader topology.
The ESM recipe selects it. Other declared tanh recipes preserve their mathematics.
A second defect prevented activation-step constants from reaching FFN dispatch.
GELU now binds the declared step on immediate and recorded execution and rejects
ambiguous steps, unknown entries, dtype mismatches, and contradictory gate layout.
Operator tests cover both precision modes, both GELU formulas, and gated execution;
binding tests exercise the actual compiled ESM recipe and both dispatch paths.

The installed candidate passes every element of all three references, then passes
again after actual device destruction, rejected execution on the lost device,
and reopening. Maximum token error is 0.0008462; observed recovery is 149.3 ms on
the same AMD device. The [final acceptance and reproduction](../reports/unfamiliar-model/20260907-esm2-8m/acceptance.json)
bind clean-checkout acceptance, successful remote CI, fresh source acquisition,
verified reconstruction, byte-identical CPU references, and installed physical
replay to commit `94570b43` and its exact runtime archive. This separate archive
is not an npm publication or qualification of the earlier pinned release.

Manual intervention consisted of choosing the checkpoint and output contract,
adapting the reusable configuration, authoring the full reference and recovery
policy, diagnosing the GELU difference, and repairing activation binding. Acquisition,
source capture, conversion, full comparison, and recovery ran as commands. This is
not yet unattended new-family automation, hardware diversity, or external adoption.
