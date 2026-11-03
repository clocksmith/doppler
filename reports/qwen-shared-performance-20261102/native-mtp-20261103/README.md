# Native MoE MTP integration diagnostic

This is a working 35B experiment and reusable GPU control/recovery implementation, not a native-MTP serving release or a speed lead. The public generic speculative path still rejects MoE. The ordinary replay path is unchanged; the rejected readback pool and checkpoint-wait candidate stay rejected/unpromoted.

## What now executes

Actual source auxiliary weights perform embedding/hidden conditioning, the learned full-attention MoE block and three GPU-chained drafts. A target pass verifies four rows; GPU argmax/greedy acceptance selects the prefix. Recovery restores initial convolution/recurrent states and reruns only their subgraphs from retained projection inputs. It does not recompute model projections or replay the model prefix. GPU selection carries the accepted hidden row and next token. Auxiliary-cache refill admits only the accepted prefix. Each consecutive-window experiment records computation, the final 36-byte feedback copy and its map in one command buffer/readback.

The component hook is explicitly attached to one recorder. Ordinary execution does not capture snapshots or create recovery closures. Parent-recorder ownership retains projections and checkpoints through deferred cleanup. A separate recovery recorder may borrow them only while that parent lease is retained. The caller remains responsible for CPU cursors, KV truncation and tentative carried tensors. This is not the full generator lifecycle integration.

## Evidence and limits

- Fourteen physical core cases pass on AMD Node and M3 Node/Metal: every rejection prefix, output caps, invalid IDs, exact recovered state, continuation and GPU-selected carry. Ownership mocks cover partial allocation/recording failure, cross-device rejection, early close and post-submit borrowing. Mocks are not GPU evidence.
- Eight real 35B cases cover all acceptance, three forced rejections, caps and post-submit restoration. Tokens match; maximum full-state difference from sequential execution is 0.00005150 and maximum logit difference is 0.00001789, below the previously frozen 0.001 bound. Full-model batched state is not bit-identical to sequential arithmetic. Lower-level retained-projection recovery is bit-identical. Initial-state restoration is bit-identical. The cancellation/callback cases invoke restoration mechanics; they do not qualify public callback handling.
- Thirty-two numerical-sequence tokens match across eight consecutive windows. All three proposals matched in those windows; forced rejection was tested separately. This is a narrow mechanics test, not useful coding quality. The manual prototype is slow: the sum of window clocks is about 2.014 seconds for 31 additional tokens, including about 0.707 seconds recording. It has no matched Pooled timing and is not promoted.
- One instrumented window attributes about 107.61 ms of compute to target layers/head, 19.40 ms to drafting, 1.64 ms to recurrent-prefix recovery and 3.34 ms to auxiliary refill. Copies and gaps are excluded. These are instrumented compute observations, not ordinary request savings.
- An isolated existing-Q6-head candidate reduced its observed compute interval from 22.15 to 8.59 ms and preserved tokens with maximum logit difference 0.00001765. It remains an unpromoted one-pair diagnostic. Target projections and recording remain concrete measured costs; checkpoint waiting has not become a proven dominant cause.

The [index](index.json) binds policies, complete receipts, scripts, snapshot identities and limitations. Files ending `.gz` retain the exact uncompressed SHA-256. The 48 log retains the initial missing-padding diagnostic failure; the fixed implementation and later physical checks pass. No tolerance was widened after observing the result. UTC execution timestamps were not recorded; monotonic clocks and the report timestamp remain distinct.

## Remaining integration

Bind native auxiliary weights and a dedicated prepared verification program in the artifact/runtime contract; wire greedy generation, accepted-prefix publication and real callback/cancellation cleanup. Reuse the base embedding/head rather than loading their duplicate component copies. Remove the measured target/recording costs before promotion, then run held-out quality and alternating complete requests against Pooled's strongest settings. Long prefill is a separate workload and still needs its own phase profile. The declared 27B, Mars and browser/Node coverage gate remains open; this experiment fills no campaign scoreboard cell.

Component: `doppler.runtime-source.gpu.kernels`. Intent: preserved. Boundary effects: explicit greedy verification, GPU token carry and native F32 recurrent-prefix recovery; no generic-MoE guard removal or default profile activation. Acceptance evidence: focused physical/ownership tests and retained diagnostics above. Source architecture retains the two existing unrelated file-size failures with unchanged budgets.
