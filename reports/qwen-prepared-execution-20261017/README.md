# Qwen prepared execution investigation — 2026-10-17

Scope: local phase tuning of Qwen3.5 0.8B and 2B Q4_K. This is not a release,
universal performance, general quality, or independent application acceptance claim.

## Decisions

- Retain immutable kernel-threshold snapshots for internal readers; the public
  copying getter retains its isolation contract. CPU decode recording improved
  in isolated trials; total decode was neutral/noisy.
- Register a separate direct-load, static-row Q4_K prefill shader and select it
  only in the named Metal profiles. Preserve original shader identities and AMD
  selection. The isolated AMD shader trial did not justify adoption there.
- Retain prepared recurrent parameters across sequence resets, clearing CPU/GPU
  convolution and recurrent state. Rebuild on source/settings incompatibility;
  unload/recovery still releases buffers. This is parameter preparation reuse,
  not reuse of private conversation state.
- Reject the vector-shared-tile and eight-row alternatives: no repeatable benefit.

## Measured outcomes

ABBA numbers below are each run's reported median. Runs differ in warmup/sample
counts; use their retained configurations, not cross-stage arithmetic.

| Experiment | Baseline | Candidate | Scope |
| --- | --- | --- | --- |
| M3 2B combined prefill, TTFT ms | 814.5 / 796.4 | 576.6 / 647.7 | threshold snapshot + direct shader, before reset reuse |
| M3 2B combined decode tok/s | 24.84 / 25.08 | 25.83 / 24.66 | neutral/noisy |
| M3 0.8B prepared reset v2, TTFT ms | 339.1 / 364.1 | 277.1 / 298.3 | 2 warmups, 3 timed |
| AMD 0.8B prepared reset v3, TTFT ms | 299.7 / 307.5 | 127.5 / 120.8 | 2 warmups, 5 timed |
| AMD 0.8B prepared reset v3, decode tok/s | 106.72 / 106.31 | 98.73 / 108.76 | unresolved variation; no decode win claim |
| AMD 0.8B prepared reset v3, complete ms | 1482.7 / 1523.6 | 1413.8 / 1286.2 | tested workload only |

`prefillRecordMs` includes awaited readbacks during preparation; it is not pure CPU
execution time. The CPU profile locates those reads in layer-state construction.
Removing them moves synchronization between phases, so submit-wait changes alone
must not be described as changes in GPU arithmetic speed.

Latest matched comparison: **Qwen3.5 0.8B**, Apple M3, 24 GiB unified memory,
macOS 26.6.2, Chrome 155.0.8059.80. Doppler Q4_K/f32 compute/f16 KV versus
Transformers.js 4.2 q4f16: comparable product formats, not identical quantized
weight bytes. Raw 64-token input, 128 generated tokens, greedy, no chat template,
2 warmups and 3 timed iterations, Transformers.js first; shared busy host.

| Metric | Doppler | Transformers.js |
| --- | ---: | ---: |
| Local model loading ms | 1981.3 | 2172.185 |
| Warm first token ms | 333.5 | 244.825 |
| Decode tok/s (127 tokens after first) | 32.15 | 24.0297 |
| Complete answer ms | 4268.5 | 5529.945 |
| Token delivery gap p95 ms | 426.3 | 57.365 |

Token agreement on this repetitive workload is useful scoped evidence, not blanket
correctness. The receipt reports claim-grade/local-comparable correctness but
`releaseClaimable: false`. Doppler still loses warm TTFT and delivery smoothness.
No fresh matched AMD or 2B comparison is supplied by this final comparison.
Loading uses local HTTP weights; OS caches were not purged. It is not a cold
network download or cold first-request measurement.

## Physical and development checks

- New shader: six shapes against an independent CPU reference and byte-exact old
  shader output on Metal and Vulkan. Final registered shader also checked on M3.
- Both models, both GPUs: four varied prompts, 32 generated tokens each, full
  reconstruction versus prepared reuse; token equality and buffer reuse. An
  aborted request followed by a new request matches fresh reconstruction.
- GPU lane: AMD Radeon 8060S, 128 GiB UMA, Chrome 146.0.7680.177, RADV Mesa 26.0.3;
  Node 22.22.1. M3 Node 22.20.0. Browser and native receipts remain separate.
- Focused threshold, dispatch, recurrent state/layout, reset, allocation cleanup,
  cancellation and embedding tests passed. Source type/style, exports, kernel
  generation/digests and routing checks passed for their scopes.
- Architecture check remains failed: unchanged attention/executors.js has 971
  lines against a reviewed 960 limit. Earlier full-suite failures/skips remain
  open. Device loss was not newly physically qualified. No Reploid deployment or
  package-release qualification is implied.

Initial physical-test attempts include a missing snapshot helper and a test that
incorrectly expected cancellation to complete silently. Corrected harnesses
accept the documented AbortError only after abort; successful complete logs are
retained beside the failures. Prepared v1 failed to cover Pipeline.reset and
showed no benefit; v2 fixes that coverage, v3 avoids per-token WeakSet lookups.

## Receipts and reproduction

- [summary.json](summary.json): 66 parsed timing receipts with source paths,
  hashes, device/memory fields and generated-token hashes. A one-token prefill
  workload has a different output hash from the 128-token workload.
- [raw-receipts.tar.gz](raw-receipts.tar.gz): 141 raw files, including command
  arguments, configurations, physical logs, comparator JSON and CPU profile.
- [source-variants.json](source-variants.json) and `source-blobs/`: exact source
  deltas against c380af74 for baseline, static-rows, direct-block, combined,
  vector-tile, prepared (final v3), and registered source. Early v1/v2 reset
  intermediates and M8 are retained observations, not reconstructed variants.
- [SHA256SUMS](SHA256SUMS): integrity of this bundle.

From this bundle directory, verify `shasum -a 256 -c SHA256SUMS`, then extract the
receipt archive into an empty scratch directory. Recorded command files contain
argument arrays; execute those arrays without shell interpolation. Replace only
machine-local model, browser, snapshot and output paths. Model artifacts must
already be provisioned. Keep workload, sampling, warmups, reset semantics,
precision and engine order explicit. Run GPU work serially.

To reconstruct an isolated historical source variant from the repository root:

```python
import hashlib, io, json, pathlib, subprocess, tarfile
bundle = pathlib.Path('reports/qwen-prepared-execution-20261017')
manifest = json.loads((bundle / 'source-variants.json').read_text())
destination = pathlib.Path('.tmp/reproduce-prepared-execution')
destination.mkdir(parents=True, exist_ok=False)
archive = subprocess.check_output(['git', 'archive', manifest['baseCommit']])
with tarfile.open(fileobj=io.BytesIO(archive)) as source:
    source.extractall(destination, filter='data')
for name, digest in manifest['variants']['prepared'].items():
    data = (bundle / 'source-blobs' / digest).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
(destination / 'node_modules').symlink_to(pathlib.Path('node_modules').resolve())
```

Use `combined` as the baseline for reset-reuse comparisons. Historical shader
trials replaced the shader in isolated snapshots; the shipped `registered`
variant instead declares a new identity and explicit Metal profile selection.
Use that variant with registered-profile command receipts. The final comparison
command is `qwen-prepared-comparison-20261017/metal-08-command.json` inside the
archive. A reconstruction does not reproduce host contention or guarantee the
same latency. Repeat both engine orders before broader publication.

## Remaining work

Trace the remaining warm TTFT and 16-token delivery bursts, preserving throughput
and cancellation. Rerun matched comparisons for both models and GPUs after a
candidate improves these phases. Measure true cold opening, memory/residency,
long prefill, quality and device-loss recovery separately. Universal competitor
superiority is not established.

Component: doppler.config, doppler.gpu, doppler.inference (prepared execution).
Intent: preserved. Acceptance evidence: scoped receipts and checks above.
Boundary effects: existing profile selection and sequence reset; no manifest
rewrite, conversation sharing, application policy or runtime default change.
