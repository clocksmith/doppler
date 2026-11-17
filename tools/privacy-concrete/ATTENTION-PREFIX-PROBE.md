# Bounded actual-checkpoint attention prefix

This measurement path is implemented and covered by control/lifecycle tests.
It has **not run encrypted Flash computation on a physical host**. Dispatch in
the unit tests is mocked. It is not a generation backend or a security-qualified
execution path; production cryptographic admission remains closed.

The fixed public fixture is `The capital of France is`, five pinned token IDs.
The existing complete checkpoint, coefficient-reference admission and checked
reconstruction backend run before private inference. The probe then follows the
ordinary graph through token embeddings, the first attention hyperconnection,
all five DeltaNet projections, attention/cache operations and residual
recombination. It stops at the input of `layers.0.mlp_hyper_connection`.

This is eight complete linear projections and 41 encrypted tiles. The complete
four-step plan still contains 603,472 tiles. Its partial route is deliberately
made unusable and closed; it is never reported as complete or as validated
generation. No MoE, later decoder, vocabulary output or token is computed.

After provider closure, separate request/key contexts execute the complete
quantized and bounded FP32 prefix references. Every output element is compared
with the quantized reference at declared absolute/relative tolerances of 0.001.
The FP32 difference is a quality observation. Only public-fixture metrics and a
tensor digest are reported; raw activations remain on the client. The policy is
immutable, its digest is bound before preparation, and a reload cannot alter
the active attempt's scope or numerical criteria.

Use the owner-confirmed resource protocol and external 8-GiB/no-swap/eight-core
limits in [BOUNDED-COEFFICIENTS.md](BOUNDED-COEFFICIENTS.md), with a fresh owned
scratch/status location. Replace `--readiness-only` with
`--attention-prefix-only --public-fixture flash-public-capital-text`. Never
reuse a previous status file or extend a resource window implicitly. The native
`.8` source is unchanged; no new native build is needed if its actual identity
still matches. Existing weights must remain in place; no new host or full-model
download is implied.

The completed readiness gate took 56.92 minutes including source verification,
complete public preparation and key/provider startup. That is measured setup
cost for a prior run, not a promised cost for this probe. Prefix evaluator,
ciphertext bytes, checked replay, full client numerical-oracle cost and memory
must be measured before any larger run. A passing prefix would still leave full
Flash encrypted output, useful client-compute savings, separate-machine
qualification and independent security review open.
