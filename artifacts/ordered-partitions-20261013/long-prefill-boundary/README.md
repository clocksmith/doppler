# Long-prefill first-boundary attribution

Component: Doppler numerical diagnosis. Intent preserved. Boundary effects: offline observation only; no changed runtime, model, reference or public package.

Reploid `2133f631` retains archive `9264d265…` failing the original 184-token prompt on Mac → AMD → Suby. Four short prefixes passed; the next prefix returned premature EOS with maximum logit error 14.471379474862829. Cleanup passed.

`compare.py` imports unchanged evaluator SHA256 `90376ddec6e6bf997425ee0e1ecf1d98533434ebbe3f934ff00a878dd49e8bc7`. Its existing layer-completion callback provides read-only snapshots; captured GPU values never enter computation. Token IDs, source configuration, pieces and manifest are bound in `proposal.json`. `observation.json.gz` decompresses to the proposal-bound JSON bytes.

The bounded AMD CPU control exits zero. Layer 7 matches the retained Mac output within 2.31e-14; layer 15 matches AMD within 3.04e-14. Independent final logits reproduce the frozen F64 reference within 3.91e-14 and select token 16. This falsifies stage-zero attribution and places the first error inside Suby output-stage execution or final readback. It does not identify a specific kernel or qualify this archive.

Command: `systemd-run --user --scope --unit=doppler-prefill-boundary-20261013 -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=0 -p RuntimeMaxSec=310s timeout -s TERM 300s env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=/tmp/doppler-reference-deps python3 /var/tmp/doppler-long-prefill-boundary-20261013/compare.py /var/tmp/doppler-long-prefill-boundary-20261013 /var/tmp/doppler-long-prefill-boundary-20261013/result.json`. Scope invocation: `bd81f96beb954c4e87bcffe24af6d7f3`. No GPU execution.
