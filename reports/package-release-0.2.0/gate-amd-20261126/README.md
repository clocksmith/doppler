# DopplerX 0.2.0, third candidate: AMD release gate (2026-11-27 UTC)

Result: **every AMD gate step passed on the exact candidate archive.** 0.2.0 matches the 0.1.7 control
on the same AMD GPU. Its tokens also match the Apple Metal gate ([gate3-20261126](../gate3-20261126/README.md))
exactly, and its protein output is within the reference tolerance. Not published; publication is the owner's step.

| Field | Value |
| --- | --- |
| Archive | `dopplerx-0.2.0-sha256-a97cb41f02e8957c9cc646ad6ff333972987c5062dbbf3e7301e947190dc97a2.tgz`, the same file the Metal gate tested |
| sha256 | `a97cb41f02e8957c9cc646ad6ff333972987c5062dbbf3e7301e947190dc97a2`, re-verified on the AMD host (install.log) |
| integrity | `sha512-6JqPVFXrS+2Q6FFqHfV0QOZtz9hwlGJCPRviE+ElFZEnpUS2qHCq4GVUIFpIv5tELEXhVJrD7x5kIK+sgXHDpQ==` (package-lock of the install) |
| Control | `dopplerx@0.1.7` from `npm pack`, sha256 `09e9ce2cc27eeacaf3702efc6320882e849d3e0ce027687ebff7f7968e30a0b0` |
| Machine | the AMD host: Radeon 8060S (Strix Halo, RDNA 3), Linux 7.0, Mesa 26.0.3 RADV (Vulkan) |
| Install | each archive by path into its own fresh directory outside any checkout; webgpu 0.4.0, doe-gpu 0.4.7, playwright 1.58.2 (no browser download) |
| Runs | 2026-11-27 00:33-00:50 UTC, one step at a time |

## Results

Same method, prompt, model and sampling as the Metal gate: `qwen-3-5-2b-q4k-ehaf16` (manifest
`a0dc69e9…`, identity digest `f959e8a03c30b822a01e4a0bff164e02cf8fa943d6614d17c3205b484a9bd90c`, recomputed
on the AMD host), "Explain in two sentences why the sky is blue.", 32 tokens, temperature 0, topK 1, chat
template. Protein: ESM-2 35M, sequence `MKTIIALSYIFCLVFADYKDDDDK`, reference
`facebook/esm2_t12_35M_UR50D@6fbf070e…` with tolerance 1e-3.

| Surface | Package | Adapter | No-fallback evidence | Tokens | Load / generate / first token (ms) |
| --- | --- | --- | --- | --- | --- |
| Node | 0.1.7 | amd rdna-3 `radeon-8060s-graphics-radv-strix-halo`, "radv: Mesa 26.0.3" | webgpu provider ok (pre-installed absent), useGPU true, `batched_gpu`, 0 fallback/CPU markers | 32 | 2148 / 994 / 243 |
| Node | 0.2.0 | same | same, `batched-gpu` | **32, identical to 0.1.7** | 2066 / 961 / 246 |
| Chrome 146 headless | 0.1.7 | amd rdna-3, "Radeon 8060S Graphics (RADV STRIX_HALO)", isFallbackAdapter false | not a software adapter by name, useGPU true, 0 markers, 0 off-machine requests | 32 | 22285 / 1754 / 1118 |
| Chrome 146 headless | 0.2.0 | same | same | **32, identical to 0.1.7 and to Node** | 21974 / 984 / 466 |

Cross-vendor: the AMD token IDs equal the Apple M3 Metal token IDs, in Node and in Chrome.

| Protein run | Artifact | Alphabet | Max abs vs PyTorch (pooled / token) | vs 0.1.7 on AMD |
| --- | --- | --- | --- | --- |
| Node 0.1.7 | 0.1.7 conversion | `amino_acid` | 4.2e-7 / 7.6e-7 | |
| Node 0.2.0 | same 0.1.7 conversion | `amino-acid` | 4.2e-7 / 7.6e-7 | bit-identical |
| Node 0.2.0 | 0.2.0's own conversion | `amino-acid` | 4.2e-7 / 7.6e-7 | bit-identical |
| Chrome 0.1.7 / 0.2.0 | 0.1.7 conversion | `amino_acid` / `amino-acid` | | bit-identical to each other and to Node on AMD |

AMD against Metal (Node, 0.2.0, same shards): pooled max abs 7.2e-7. The token-embedding digests differ
by vendor (AMD `e76a198d…`, Metal `33645eae…`), so the two vendors' floats are close but not bit-equal.

ESM-2 artifact: the Mac conversion from the Metal gate no longer existed, so the official files were fetched
again at revision `6fbf070e…`. Their sha256 values match the reference fixture. They were converted on the Mac
with each package's own converter and config (`esm2-convert-017.log`, `esm2-convert-020.log`). The shards and
tokenizer are byte-identical to the Metal gate's (`7d0c833b…`, `42cfb7b6…`, `f874eed6…`). The manifests differ
from it only in `metadata.convertedAt`, so the identity digests differ: 0.1.7 conversion `9de36f9b…`,
0.2.0 conversion `2bc8c27b…`. Shards were checked by Doppler's own loader verification (`shardHashCheck: doppler`).

## Host conditions

Each GPU step ran through `run-guarded.sh`, which runs `precheck.sh` immediately before the step. The precheck
requires: physical-disk I/O of at most 20 MiB/s and no copy, scan or hash process; zero system and user-slice
memory pressure; MemAvailable stable over 60 s; nothing holding `/dev/kfd`; only the desktop and the standing
participant holding the DRM device; no exclusive holder of the fleet GPU lock (the step held it shared); no task
in D state; and the standing public participant idle, meaning its last partition-state event was not
`executing`. Every step passed (`*.precheck.log`): load 1.5-2.5, MemAvailable 82-83 GiB and flat, pressure 0.00.
The watchdog (`*.guard.log`) recorded peak process-tree RSS of 62 MiB to 1.5 GiB, no SIGTERM, no kernel
amdgpu/drm messages, and no participant request during any step. The participant kept running untouched.

Deviations:

- The first 0.2.0 Node text attempt (00:35 UTC) was refused by its precheck: residual memory pressure of 0.26
  from the 0.1.7 step a minute earlier (that step briefly reached 1.08 while allocating GTT). The step was rerun
  at 00:38 UTC once pressure read zero. Each later step first waited for zero pressure.
- Another lane's browser job ran on the host throughout. On the owner's instruction, an active other-lane
  process was recorded rather than treated as a failure. The job rendered with SwiftShader on the CPU and held
  no GPU device (precheck holder lists). For this reason the timings are informational; correctness is the gate.
- `browser-gpu-gate.js` adds the Linux Vulkan flags (`--use-angle=vulkan --enable-features=Vulkan
  --disable-vulkan-surface`) and a by-name software-adapter rejection. `node-gate.js` is unchanged from the
  Metal gate. The Metal gate used Chrome 158; this host has Chrome 146.
- Committed scripts write the host's gate directory and GPU lock path as `$GATE` and `$GPU_LOCK`.

Install directories, models and processes were removed from the AMD host afterwards.
