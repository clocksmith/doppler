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
