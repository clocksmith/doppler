# Long-prefill first-boundary attribution

Component: Doppler numerical diagnosis. Intent preserved. Boundary effects: offline observation only; no changed runtime, model, reference or public package.

Reploid `2133f631` retains archive `9264d265…` failing the original 184-token prompt on Mac → AMD → Suby. Four short prefixes passed; the next prefix returned premature EOS with maximum logit error 14.471379474862829. Cleanup passed.

`compare.py` imports unchanged evaluator SHA256 `90376ddec6e6bf997425ee0e1ecf1d98533434ebbe3f934ff00a878dd49e8bc7`. Its existing layer-completion callback provides read-only snapshots; captured GPU values never enter computation. Token IDs, source configuration, pieces and manifest are bound in `proposal.json`. `observation.json.gz` decompresses to the proposal-bound JSON bytes.

The bounded AMD CPU control exits zero. Layer 7 matches the retained Mac output within 2.31e-14; layer 15 matches AMD within 3.04e-14. Independent final logits reproduce the frozen F64 reference within 3.91e-14 and select token 16. This falsifies stage-zero attribution and places the first error inside Suby output-stage execution or final readback. It does not identify a specific kernel or qualify this archive.

Command: `systemd-run --user --scope --unit=doppler-prefill-boundary-20261013 -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=0 -p RuntimeMaxSec=310s timeout -s TERM 300s env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=/tmp/doppler-reference-deps python3 /var/tmp/doppler-long-prefill-boundary-20261013/compare.py /var/tmp/doppler-long-prefill-boundary-20261013 /var/tmp/doppler-long-prefill-boundary-20261013/result.json`. Scope invocation: `bd81f96beb954c4e87bcffe24af6d7f3`. No GPU execution.

## Installed output-stage observations

`diagnostic-attempts.json` retains every failed or rejected observer attempt. Before-completion copies and unavailable debugger closures are not kernel evidence. Two controls exceeded the unchanged 10-second attempt-close deadline; later outer resident/provider closure passed, which does not erase those failed controls.

`canonical-layer-capture.json.gz` contains installed operator captures from fresh Suby resident invocation `ac78cd2bc65f4283a3e94f7e2f557af8`, original 184-token output-stage input and unchanged archive. Nine distinct probe targets emit seventeen records because step and final layer outputs share names. The diagnostic cardinality assertion fails; all tensor bytes and failure remain retained. Layer 16 input normalization matches within 3.85e-13 and both layer-16 outputs within 4.67e-14. Layer 17 is the first divergent layer; its two output snapshots differ. This narrows attribution, not root cause. The fresh-resident token is 1, unlike the original campaign's EOS; this is not identical-run reproduction.

`canonical-memory.jsonl` records the bounded native provider exit. Outer resident/provider closure passes. No changed runtime bytes or new qualification threshold. `independent-layers.npz` stores unchanged evaluator outputs 16–23 from a separate bounded CPU control; `independent-layer-15.npy` binds to `result.json` for normalization comparison. `compare-stage.py` validates shape, byte length, storage digest, independent tensor hashes and original weight-shard identity before comparison.

Replay the offline comparison by decompressing `canonical-layer-capture.json.gz` to `stage-result.json`, then running `python3 compare-stage.py <this-directory>` with the manifest-bound model at its recorded path. `stage-inputs.json.gz` and `isolated-stage-input.json.gz` retain original inputs; decompress before running their diagnostic controllers. Manual observer variants are retained rejected approaches, not runtime implementations. `recover-canonical-capture.js` attempted a compact export after model completion, but the provider had already closed; its connection reset performed no export.

## Layer-17 intermediate control

`layer17-result.json.gz`, `layer17-captures.json.gz`, and `layer17-comparison.json` retain scope `53991e149a984bf7a90ee54165b1e825`. Seven snapshots completed through the installed canonical observer. Normalization first differs at row 32 (rows 0–31 match below 2e-11); downstream projection/residual outputs contain 182 zero rows. Token zero differs from the original full campaign, so attribution remains diagnostic. Preparation, execution observation, compressed export, attempt close, resident close and provider close all complete within unchanged deadlines. `layer17-memory.jsonl` retains host accounting and provider exit.

`reference-layer17.py` evaluates the original equations using independent layer-16 state and verified original weight shards. Local NumPy 2.2.3 differs from the retained AMD NumPy 2.4.4 output by 3.11e-15. An initial exact-byte assertion rejected this tiny CPU difference; the report now records it under the existing 2e-11 operator diagnostic tolerance, without altering the frozen 0.001 full-model acceptance reference. The next discriminator is layer-17 input versus normalization output.

## Input-boundary control supersedes fixed layer attribution

`input17-result.json.gz` and `input17-captures.json.gz` retain invocation `82cf27847a174af6a8222e2e9c5f9fca`, with original input, unchanged archive and declared immediate prefill. The preceding recorded-only guard rejected observation; its separate result and memory receipt remain retained. This policy was not changed to force recorded execution.

`input17-comparison.json` shows layer 17 receives bytes identical to the second layer-16 output snapshot. `input17-output-mutation.json` records 188,412 differing logical values between the two layer-16 output captures, maximum difference 2.393663564746234; the second contains 182 zero rows. Unlike the earlier control, layer 16 already diverges. These timing-sensitive observations do not establish a normalization root cause. The next diagnostic must bind buffer identity and slot ownership before attributing the changed bytes to a particular operation or release.

Preparation, observed execution, export, attempt/resident/provider close passed; `input17-immediate-memory.jsonl` records provider exit zero and no OOM. These are diagnostic results, not numerical acceptance. Neither the original 0.001 tolerance nor public runtime changed.

## Buffer identity control

`ownership-result.json.gz` and `ownership-captures.json.gz` retain invocation `210b4c33cf3b4871b8432e74673af6c2`. The two layer-16 captures bind the same GPUBuffer and reproduce distinct snapshots. The layer-plan slot breakpoints produced no events: this execution uses the standard layer/FFN path. The previous description of the interval as slot cleanup was an unproven assumption; the actual interval crosses standard FFN cleanup. `ownership-comparison.json` records this limitation rather than claiming the output was released. Preparation, observed execution, attempt/resident/provider closure passed; provider exited zero, host peak 1,288,044,544 bytes, no OOM.

## Standard FFN cleanup control

`ffn-ownership-result.json.gz` and `ffn-ownership-captures.json.gz` retain invocation `c890c271ce974d3eabfea1617cfc32ec`. Canonical snapshots still change across standard FFN cleanup. Its returned output is buffer 1; cleanup releases distinct buffers 2, 3, 5 and 4. No release of output 1 occurs between its two captures. `ffn-ownership-comparison.json` therefore falsifies premature output release in this control. Later releases/reacquisitions belong to subsequent work and do not prove an earlier use-after-free. Readback/provider behavior versus earlier model dispatch remains unresolved. Attempt/resident/provider close passed; provider exit zero, host peak 1,282,273,280 bytes, no OOM.

## Isolated immutable readback

`readback-control.js` opens the same installed native Vulkan provider and verified 1,984-file runtime, uploads the original 1,507,328 activation bytes without model dispatch, and uses the canonical snapshot API eight times. `readback-bound-result.json` records all eight storage digests equal to the original, queue settlement and host close. This falsifies corruption caused by isolated copying/readback/staging retirement; it does not rule out an interaction with model dispatch or allocation retirement. The next discriminator is a minimal paired compute result captured repeatedly with and without intervening temporary-buffer retirement, under the same provider.

The first launcher lacked its remote input and never opened a provider. The next omitted the debug module's required device binding and rejected snapshot acquisition; `readback-unbound-control.js` and `readback-unbound-result.json` retain that setup failure. The corrected observer binds the same owned device and clears it on close. These controls do not modify the archive, model arithmetic, reference, or release acceptance.

## Dispatch lifetime and driver fence attribution

The paired-retirement, paired-wrapper, paired-chain and paired-gc controls each
pass with eight identical canonical snapshots and confirmed provider close.
They cover temporary-buffer retirement, installed wrapper uniform retirement,
64 queued paired additions with pool reuse, and forced JavaScript collection.
They do not reproduce the full model's failure.

`dispatch-lifetime-comparison.json` binds both layer-16 snapshots to the same
output buffer. The intervening events are staging allocation, readback
submission and staging destruction, with no observed output release or compute
submission. The separate uniform-retention treatment holds 141 uniforms until
operation completion; corruption persists. Both full-stage runs return token
zero and fail numerical acceptance despite successful host closure.

`dispatch-lifetime-kernel.txt` records i915 fence-expiration timeouts for the
exact owned provider PIDs 1099138 and 1102755 during those two runs. WebGPU did
not report a validation error. These kernel failures supersede an attribution
based solely on changing snapshots: the observations do not prove a standalone
readback or lifetime defect. Do not repeat the unchanged full-stage workload.
Compressed result/capture receipts and host-memory samples preserve both runs.

`linear-phase-control.js` uses the original installed nine paired linear shaders
at 16 tokens, 16 key/value heads and 128-dimensional heads. Each phase is
submitted and settled independently, an explicit diagnostic scheduling change.
Independent F64 equations pass for convolution, convolution state, recurrent
state, gates and output under the existing 2e-11 operator threshold. Recurrent
state takes approximately 298 ms in this control. This is operator evidence,
not the 184-token model case or the unchanged 0.001 release gate.

The next discriminator is bounded projection dispatch timing versus cumulative
queued work. The eight-row matmul diagnostic uses independently constructed
F32/Q4K weights, the installed shader and independent F64 dot products. Its
reservation caps host memory at 2 GiB and execution at 180 seconds. Preparation,
execution, numerical comparison and settlement must be reported separately.
