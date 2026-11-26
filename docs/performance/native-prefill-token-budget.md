# Native prefill token budget diagnosis

Component: `doppler.runtime-source.config`
Intent: preserved. Boundary effects: native experimental runtime profile and tests.

The failed native diagnostic used source 5d2c13bf, native preserved GGUF blocks,
F32 activations/KV and actual READY `prefillTokenChunkSize:null`. The product
control explicitly used 256. `prefill-policy.js` returns null when effective
chunking is disabled; `prefill-runtime.js` then calls `_prefill` with all 2287
prompt IDs. `prefillChunkLayers:4` limits layers per recorder submission; it
does not split token work. There is no automatic token cap in this path.

The failed request returned zero tokens after 2283.1 ms. Log 350033.9 ms is a page
logger timestamp since startup, not request duration. Kernel journal 00:17:34 UTC
records a gfx ring timeout in the diagnostic Chrome GPU process and automatic
recovery. The following allocator/live-device error at linear input norm is a
consequence of device loss, not proof that RMSNorm caused the failure. A last
submitted kernel identity was not captured; no specific shader is established
as the timeout cause.

The native profile inherited disabled token chunking. This change declares 256
using the existing session field. It introduces no hidden fallback, new time
threshold, shader substitution, dtype change or generic device default. It
retains the profile's experimental status. A benchmark overlay must also set
256 explicitly and reject a READY receipt that resolves differently. The
frozen failed diagnostic remains evidence and must not be overwritten.

CPU regression calls the production token planner and prefill adapter with a
mocked execution boundary. Before this change it fails on inherited null;
afterward it observes 8×256+239, and only the final chunk computes logits.
Short inputs remain one invocation; 257 becomes 256+1. Existing batching-policy
checks remain intact: token chunking currently disables command batching, so
this is also a submission-path difference, not only a loop-bound change.

A reduced-head physical continuation fixture uses native GGUF source layout,
F32 state, head width 128, one key head/two value heads and convolution width 4.
On Apple M3 Metal Node 22.20 / webgpu 0.4.0, whole 257 vs 256+1 produces identical
outputs and full convolution/recurrent state (maxAbs 0). This is bounded
component evidence, not a 35B model run, AMD/Chrome qualification, performance
result or proof of watchdog causality. The fixture allocates the full
convDim×kernelSize state, including the newest slot. An earlier incorrectly
sized fixture receipt is retained as invalid and excluded.

Pending: native full-model numerical/quality and long-request qualification on
amd-host Chrome with actual READY admission. No unbounded 2287-token GPU replay
is required to establish the configuration mismatch.
