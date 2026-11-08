# Demo activation recording and live precision comparison

Physical Apple M3 / Chrome WebGPU check, explicit command:
`node tools/experiments/activation-grid-browser.js --physical`.
[Receipt](physical-browser.json) records the repaired, pinned Qwen 3.5 0.8B artifact.
Greedy generation with and without activation recording produced identical token
IDs and observed token records (including logits/probabilities). All 72 layer
captures completed, with both prefill and decode, and zero dropped captures.
Geometry is 24 layers × 1,024 values; RGB packs three consecutive numerical values.
This is layer-output activation capture, not weights, KV or recurrent-state capture.

[Standalone recording](recording.json.gz) decompresses to importable JSON.
[Integrity](recording-integrity.json) binds the complete numerical payload, layout,
fixed scale, model/build identity, capture limits and token/phase provenance.
The browser unloaded the model before importing/rendering the recording. The grid
is a labeled composite of sequential layer captures for one forward/input position;
it does not assert simultaneous state. Readback is asynchronous and bounded;
limits stop recording, not generation. Backpressure/failure markers are explicit.
Timings under observation are diagnostic, not representative throughput.

[Live comparison](precision-comparison.json) uses the arbitrary composer prompt
`What is 2 plus 2? Answer briefly.`, the same artifact and exact prompt tokens,
greedy seed 0 and a 16-token cap. Both paths generated `4` and stop. The candidate
requests the existing selective-F16 transform: residual activations remain F32.
The two resolved execution identities differ and both receipts are retained.
This is not full-F16 inference, a benchmark or a quality qualification. Later
output after a divergent token also includes changed context. Unsupported paths
fail visibly and the temporary model is unloaded in all cases.

The comparison exposed a production handoff bug: a legacy F16 lane request was
rejected after the compiler resolved its selective-F16/F32-residual contract.
Only a compiled explicit selective lane consumes that request; canonical session
activation, KV and output dtype mismatches remain errors. The focused regression
exercises both boundaries and the rejection cases.

[Browser controls](browser-controls.log) cover standalone import/export, playback,
scrubbing, failed-import preservation, unrelated-receipt separation, shuffled
examples, visible model selection, continuous token gradients and aligned bars
at 1,200, 390 and 320 pixels. These UI checks use synthetic data/mocked generation;
they are separate from physical capture evidence. [Grid](physical-grid.png) and
[phone detail](token-inspector-390.png) were visually inspected.

The shell adds four JS modules to the audited 824-module runtime baseline:
three activation modules and one live comparison module, 828 total. Its existing
ten-megabyte byte limit is unchanged. New implementation files are strict typing
and unused-local roots. [Focused tests](focused-tests.log), [types](typecheck.log).
