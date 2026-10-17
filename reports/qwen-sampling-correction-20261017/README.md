# Qwen sampling correction and frozen candidate — 2026-10-17

Candidate archive SHA-256:
`75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2`.
Runtime source commit: `bcb6c045`. This is a qualification candidate, not a release.

The [previous archive](../qwen-package-candidate-20261017/README.md) remains failed
on Metal sampling. Atomic scratch access repairs the observed heap corruption;
the independent reference and acceptance thresholds are unchanged. M3 and AMD
pass 540 immediate/recorded F32/F16 sampling cases and the browser seeded batch
audit, including split/retry equivalence, cancellation, consumer callback failure,
and settled transient resources. These are operator/lifecycle checks, not blanket
model correctness.

Both Qwen 0.8B and 2B pass prepared reset and cancellation checks on both hosts.
Browser compatibility-export journeys on both GPUs pass public logits, cancellation,
reuse, device destruction after completed work and immediately after an observed
submission, rejected stale-session execution, explicit unload, and device recreation
with identical text. Deliberate destruction does not prove recovery from every
spontaneous driver failure.

The exact archive passes installed imports, exports/types, dependency/closure,
synthetic consumer, and package checks. Its first installed attempt failed on a
stale generated runtime-closure receipt after registry synchronization; refreshing
that receipt and verifying the **same archive** passes. Initial receipt retained.
Size: 1,959 files; 3,483,987 packed bytes; 14,168,991 unpacked bytes. All original
limits remain unchanged.

Matched comparison evidence is produced by [the retained driver](run-matched.py)
and [deterministic summary](summarize-matched.py). See [all measurements](comparison-table.md)
and [machine-readable summary](comparison-summary.json); the summary's `complete`
field, individual status, and fairness verdicts own completion. Each GPU runs both
Qwen3.5 sizes, both engine orders, 64-token cold/warm input, and 1,024-token warm
input, with 128 greedy output tokens. Warm runs have two warmups and three measured
requests; cold runs have one model opening followed by three measured requests.
No outliers are removed. The retained final transcript matches on completed runs;
this is workload evidence, not general model correctness or task quality.

Hardware is Apple M3 / Metal (24 GiB, Chrome 155.0.8059.80) and AMD Radeon 8060S /
RADV STRIX HALO / Vulkan (Chrome 146.0.7680.177). Engine comparisons stay within each
host. Transformers.js 4.2.0 uses pinned `onnx-community/Qwen3.5-0.8B-ONNX-OPT` and
`onnx-community/Qwen3.5-2B-ONNX-OPT` artifacts, `q4f16`; Doppler uses
`qwen-3-5-{0-8b,2b}-q4k-ehaf16`, Q4_K weights, F32 compute and F16 KV.
These are disclosed optimized-format comparisons, **not identical quantizers**.
Each receipt binds its comparator revision, prompt, sampling, model manifest,
resolved execution, browser, hardware, and fairness gate.

Cold means engine profile/cache cold with provisioned local weights. OS/driver
caches are not purged and there is no network download. The original engine
`timing`/`firstLoad` summaries mix the cold opening with a median resident request;
the derived `coldOpening` table instead uses each engine's actual first sample.
There are two independent cold openings per model/GPU, one per engine order.
Opening starts at model initialization, not browser/process launch. Missing
whole-application startup measurements remain unavailable. Warm TTFT, internal
prefill, decode and complete generation wall time remain separate in the JSON.

AMD's primary driver predates the unselected-profile loading correction; M3's
includes it. That change only prevents loading irrelevant profiles before a run;
selected execution and immutable package bytes are identical. Driver hashes are
retained. Hosts were not exclusive, and M3 2B shows substantial variation; do not
turn a favorable order into a general ranking. Failed setup attempts are retained
alongside completed runs, never counted as performance samples.

All 24 primary pairs completed with unchanged fairness gates and matching retained
transcripts. All four separate M3 memory pairs also passed those gates.

The [memory measurements](memory-table.md) and [raw summaries](memory-summary.json)
keep sampling limits explicit. The separate `--memory-only` M3 pass uses driver commit `3ef32753` and the same
immutable runtime. `/bin/ps` samples the engine child process tree every 500 ms;
aggregate RSS includes shared mappings and can miss shorter peaks. These samples
do not replace primary latency numbers. Linux procfs/rocm-smi evidence remains in
the AMD primary receipts. GPU vendor counters are device-global; Doppler's pool/KV
allocations and Transformers.js's JavaScript heap are different surfaces. Neither
establishes an apples-to-apples dedicated VRAM win. CPU counters cover observed
processes and may undercount processes that exit between samples.

Raw logs, exact commands, failed attempts, physical receipts and comparison
receipts are retained in `raw-evidence.tar.gz`; extract it in this directory to
restore the relative paths in the summary. `SHA256SUMS` binds the archive and
readable reports.

Reproduction: provision the receipt-pinned models and existing dependencies, then
use `prepare-harness.py REPO ARCHIVE DRIVER_COMMIT NEW_DIRECTORY` to extract the
exact npm runtime and isolate the three repository-only URL helpers outside
`src`. Run `run-matched.py SNAPSHOT MODEL_ROOT COMPARATOR_RELAY CHROME OUTPUT
SNAPSHOT/archive-files.json`. Every paired run rechecks all 1,959 archive files.
Use a fresh output directory; failed attempts are not overwritten. Append
`--memory-only` for the separate four-pair macOS memory pass. Commands and source
hashes are retained with every pair. The primary M3 and AMD driver snapshots are
also represented by their committed code and recorded helper-import rewrite.

CI remains open: `39069960131` (`4a38847d`) and `39071539236` (`a19370d4`) fail
`tests/training/native-lora-round-gpu.test.js` because the hosted runner has no
usable WebGPU adapter. Intermediate runs were cancelled, not passed. The
[CI snapshot](ci-status.json) and raw failed logs retain those outcomes. This cycle
does not expand into LoRA work or hide unrelated skips. The Apple-specific kernel
test remains explicitly unsupported on AMD; it executes successfully on M3.

Component: `doppler.repository-tooling`, `doppler.tests`, kernel execution.
Intent: preserved. Acceptance evidence: exact archive package/consumer receipts,
scoped numerical and lifecycle results, matched comparison receipts. Boundary
effects: none; no release, catalog, production setting, or application authority
changed. Delivery attribution and per-tensor reachability are the next steps,
in that order; reusable bindings/scratch follow their measurement. No WAN,
LoRA, prefix-cache or new-model work is part of this cycle.
