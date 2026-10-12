# Resident prefill submission-policy repair

The failed archive9 split sends184rows through one layer12 encoder with14dispatches. Ordinary prefill under the same explicit prefillTokenChunkSize256 calls shouldDisablePrefillCommandBatching and uses immediate operators. The resident path ignored that policy even for prompts shorter than the configured chunk. Both retain one-layer carry boundaries.

The repair consumes the existing decision for resident embedding/layers. A separate boundary recorder still owns carry copies and confirmed-completion boundaries, while direct-mode buffers are tracked independently. Unknown queue completion moves those owners (including cancelled logits) to the existing pool deferred-destruction path, never reusable pooling. The original error remains first in an aggregate when completion also fails. Public math, shader bytes, row count, weights, source precision and reference are unchanged.

The existing partition-prefill-lifetime test fails before the routing repair and passes after it. It covers recorded/direct modes, prompts below configured token chunk size, carry ownership, layer/copy/submission/recorder failures, cancellation, paired byte rows, rejected completion and cancelled-logit quarantine. A real BufferPool/device-budget control verifies that rejected queue completion retains the device charge and rejects new allocations; destruction/reuse becomes possible only after explicit successful settlement. Mocked orchestration is not physical GPU proof.

Component: doppler.runtime-source.inference.pipelines.text
Intent: preserved
Acceptance evidence: retained focused test, source type/style/architecture logs; bounded physical replay still pending.
Boundary effects: resident partition submission now consumes the existing whole-model prefill batching policy; no transport or admission change.
