# Rig TargetPlan paired storage sizing

The actual paired execution identity declares `intermediatePrecision: two-f32` and `linearRecurrentStatePrecision: two-f32`. Rig previously sized hidden activations, recurrent state and convolution state as one F32 component despite retaining that identity in the signed plan. The failing regression reproduces a 20,480-byte hidden row where the declared representation requires 40,960 bytes.

Specialization now reads the existing observed program-load session. Hidden and convolution storage use the intermediate component count; recurrent storage uses its independently declared state component count. Source ModelIR remains unchanged. KV and public logits sizes are unchanged. Unsupported or missing precision in an existing signed policy, paired storage with F16 source values, and paired intermediates with scalar recurrent state reject before signing. The pre-existing scalar path without a signed program-load policy retains its prior one-component representation.

`source-memory-review.json` calculates the actual 0.8B source-component slot deltas from the retained ModelIR and observed initial identity. These are declared slot sizes, not measured allocation peaks or a complete lifetime analysis.

Validation: converter stage and signed-tooling tests pass, including scalar and paired sizing, state-only pairing, unchanged KV/logits, source preservation and mismatch rejection. Source type, style and architecture checks pass. The mistaken initial typecheck command is retained separately. This is a Rig-only source correction; the already-qualified standard archive `102fd0c4…` was not repacked or altered.

Component: doppler.runtime-source.converter. Intent: preserved. Boundary effects: signed TargetPlan memory expressions accurately describe existing paired storage.
