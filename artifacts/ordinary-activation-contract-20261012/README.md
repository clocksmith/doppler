# Ordinary activation metadata

Component: doppler.runtime-source.inference. Intent preserved: explicit model/activation contracts; applications own transport and publication.

The existing speculative block validator requires a non-null parent and proposal. That cannot honestly represent the first ordinary prefill. The new public `validateOrdinaryActivation` explicitly admits `doppler.ordinary-activation/v1`, with `proposalId:null`, a null parent and row zero only for sequence zero, and one absolute input row with a parent for decode. It shares the existing exact scalar/F16/two-F32 layout, digest and copied-byte validation. Existing block-v2 validation remains separate and unchanged in behavior.

`before.log` retains the missing public method. `ordinary.log` covers all three layouts, immutable metadata/bytes, both phases, unknown fields/versions, missing values, trailing bytes, payload tampering, row/stride/component mismatch and overflow. `block-v2.log` reruns the existing unmodified validator test. Source-bound successful type/style/architecture checks and one intermediate validation-order failure disposition are recorded in `checks.json`.

No model computation, installed archive, accepted pin or physical qualification changes here. Reploid must explicitly require this capability before using the new metadata.
