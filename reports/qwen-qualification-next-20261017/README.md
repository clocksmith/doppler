# Qwen qualification checkpoint — 2026-10-17

Qualification remains incomplete. This checkpoint repairs source/test gates; it
makes no new performance, numerical, cross-device, or release claim.

## Changes

- `0a304fad`: move online attention pipeline-constant resolution into the existing
  attention planner with explicit policy input. The executor returns below its
  reviewed architecture ceiling; no architecture budget increased. Remove unused
  imports, expand unused-local checking, and ratchet declaration debt downward.
- `3c488a92`: repair thirteen failing unit files without changing production model
  settings. Fixtures now declare required runtime pipeline/speculation fields;
  Qwen comparator references and their generated quickstart mirror match the
  catalog. Local cache snapshots are excluded from component discovery.
- Historical conversion tests explicitly account for source changes `17dc3b3a`
  and `56e3abbd`, preserving archived fixture hashes and artifact identities.
  Structural generation checks follow the shared helpers; behavioral tests exercise
  fallback, unrelated failure, diagnostics, and cleanup through all three public
  generation surfaces.
- The W8A8 exporter now archives the exact operand-source bytes matching its
  retained physical receipt, recovered from `1168be7c`. The old receipt, oracle,
  fixture, and shader hashes remain enforced. This is an offline export repair,
  not a new physical qualification.
- Regenerate package exclusions, runtime closure, and dependency views from their
  owning tools. Removed exclusions refer only to files no longer on disk.

## Evidence and remaining gates

The complete unit rerun exits successfully across 946 discovered files, with
16 files reporting at least one skipped check. The [machine status](status.json) records
executed checks and skips. Raw logs and
package inventories are retained in [the evidence archive](raw-evidence.tar.gz),
with [SHA-256 checksums](SHA256SUMS). The initial unit failure log is preserved,
including all thirteen failures; focused reruns are separate receipts.

The repository check chain was run with `test:unit` separately. Initial failures
were stale runtime closure, stale package exclusions, installed closure mismatch,
and stale dependency views. Regeneration and reruns repair those stale views and
installed-package checks, but expose an existing **package budget failure**:

| Measure | Measured | Existing limit |
| --- | ---: | ---: |
| Shipped files | 2,024 | 2,004 |
| Packed bytes | 3,503,884 | 3,497,999 |
| Unpacked bytes | 14,265,879 | 14,172,897 |

The [inventory delta](package-delta.json) reproduces the last 2,004-file baseline
at `11d941d41` with the same local npm toolchain. Twenty added files include
Qwen profiles, prepared execution, paired shaders, shared helpers, and previously
added training code. No files were removed. Limits are unchanged; this gate
remains failed. Installed package smoke passing does not waive that failure.

Unit-runner success does not close self-skipped physical/artifact checks,
excluded GPU/browser suites, or the registered pending public-logits test. The
status file keeps those distinctions. Earlier CI failures remain historical
failures; no CI completion is inferred from local checks or canceled runs.

## Resume in order

1. Resolve the package budget failure without removing required runtime assets or
   raising acceptance limits merely for green. Recheck `npm run public:boundaries:check`.
   The other stale-view checks have their own passing reruns.
2. Execute relevant skipped Qwen checks using provisioned models, then qualify the
   exact candidate on M3/Metal and AMD/Vulkan. Alternate engine order for 0.8B and
   2B. Separate process/cache-cold opening, warm TTFT, long prefill, decode, memory,
   cancellation/reuse, and device-loss/recreation. Do not call unpurged host caches
   network-cold or claim process GPU residency from device-global telemetry.
3. Attribute the known 16-token delivery bursts across GPU submission, readback,
   sampling, enqueueing, and consumer delivery. Preserve measured decode throughput.
4. Implement immutable per-tensor/alias packed/dense/both reachability including
   fallback paths; measure 27B Q4 and 2B before reusable bind groups and scratch
   arenas. Prefix checkpoints follow. No WAN or LoRA work in this cycle.

The last matched performance evidence remains
[the prior comparison](../qwen-profile-correction-20261017/README.md): it contains
wins and losses and does not qualify this checkpoint's current source on either GPU.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.tests`,
`doppler.repository-tooling`, `doppler.runtime-source.config`,
`doppler.benchmarks.vendors`, `doppler.docs`.
Intent: preserved. Boundary effects: none; no application or runtime authority
changes. Acceptance evidence: focused reruns, complete unit-runner receipt,
repository checks and explicit unresolved package/physical gates described above.
