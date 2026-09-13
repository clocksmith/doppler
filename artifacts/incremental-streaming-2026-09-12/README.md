# Installed streaming and shared-resource acceptance

This record covers September 12-13. The candidate adds opt-in operation v2,
incremental Unicode decoding and consumer reconstruction, native Reploid journal
appends, and focused lifecycle repairs. Existing v1 behavior remains available.
The candidate is unpublished; no deployment was performed.

## One immutable candidate

- Doppler runtime source: `54299e47484a71faeebdbb6bbd7a07a363157816`.
- Reploid consumer: `46a27ee175086e68fbf2603a13a3bc7f534609e6`.
- Archive SHA-256: `c21230a20471c8beb9fa414a326b7a2da889a4ef9d07c13f3f23e8db9a1ca962`.
- Archive integrity: `sha512-rdesSSP6rFYmgf+pbIobrH99hs+HaAJfv7/lWl9FkGldlVOcdmSC7ZOHZhu+MxNBtmfeW2/AhWIUBVWbbHiSVg==`.
- Package version: `0.6.1`; 1,789 files, 2,119,045 packed bytes, 10,906,744 unpacked bytes.
- Local and CI tarballs are byte-for-byte identical. Node 22.23.2 and npm 10.9.8
  reproduce the archive. The CI fixture revision is
  `96776029b91895eba882c7803ba50b0fb58ebfbf`.
- [Download the retained CI artifact](https://github.com/clocksmith/doppler/actions/runs/34737393783/artifacts/10311775717).
  [Archive identity and CI record](final-ci-record.json) binds its ZIP digest.

Published npm 0.6.1 is a different archive, SHA-256
`96d2699a3e677815890f804459c023bea4fb2e2a1a59d157b5b6a04c9e573d5f`.
Its completed installed baseline remains in
[the baseline record](../installed-consumers-2026-09-12/). Reploid's package and
browser defaults now pin those published bytes. V2 requires explicit adoption
of the candidate; the version string alone does not identify these changes.

## Installed consumers and transport

[The required cross-repository job passed](https://github.com/clocksmith/doppler/actions/runs/34737393783).
Missing installation or fixtures fail the job. Generated signed fixtures remain
outside the production package.

- [Standalone](final-ci-installed-consumer.json): public imports and declarations,
  generation, embedding, reranking, direct and HTTP streaming, cancellation and
  browser-conditioned host exports.
- [Reploid runtime](final-ci-reploid-consumer.json): resolved generation settings,
  actual stop reasons, cancellation, stale requests and request-bound adapters.
- [Actual WebRTC and native IndexedDB](final-ci-webrtc-consumer.json): three v2
  partials, two deliveries, one model execution. A replaced provider replays the
  saved completion; the requester reconstructs and independently verifies it.

These CI checks use injected model programs. They establish installed API and
transport behavior, not physical model qualification or independent operators.

## Output-processing measurement

[The synthetic measurement](transport.json) uses the actual bundled tokenizer,
operation adapters, executor, JSON transport and consumer helper.

| Tokens | V2 transferred bytes |
| ---: | ---: |
| 1,024 | 472,285 |
| 2,048 | 944,349 |
| 4,096 | 1,888,477 |

V1 transfers 27,142,194 bytes at 4,096 tokens. V2 processes 4,096 incremental
token IDs; v1 copies 8,390,656 IDs into partial payloads. CPU measurements split
producer, serialization and consumer parsing/verification. Logical payload counts
are not a count of every engine copy, and V8 allocation samples are estimates.
No GPU inference speedup is claimed. The separate native-journal logs show
approximately linear writes and reads after removing cumulative rewrites.

Unicode, malformed byte boundaries, split stops, slow consumers, cancellation,
output limits, interrupted streams and final integrity have focused regressions.
Applications append valid display deltas and can batch DOM writes by animation
frame. A raw token need not produce a display character. Snapshot materialization
is an explicit compatibility cost.

## Lifecycle findings

Injected failure tests exposed stale results after device loss, admission after
loss during program loading, reuse of a destroyed slot after failed allocation,
and incomplete cleanup after a release hook throws. The repairs preserve explicit
ownership and asynchronous draining; cancellation does not preempt submitted GPU work.

The [physical two-session failure](physical-shared-device-failure.json) found an
additional composed bug: restoring the first session rebound the same GPU device
but advanced its generation. The global pool then destroyed the second session's
live weights and RoPE buffers. Its next embedding was all zeros, with explicit
WebGPU destroyed-buffer errors. [The failing deterministic reproduction](shared-pool-before.log)
and [passing regressions](shared-pool-regressions.log) localize this to device
generation, without a new ownership framework. The fix advances generation only
when the physical device changes.

The [standalone physical run](final-physical-standalone.json) passes generation,
embedding and reranking in Chrome 146.0.7680.177 on AMD RDNA 3, without a fallback
adapter. Qwen3-4B-Instruct-2507 matches all 16 frozen reference token IDs and
stops at the token limit. Qwen3-Embedding-0.6B returns two finite 1,024-dimensional
vectors. Qwen3-Reranker-0.6B completes its two-document operation; no new semantic
ranking oracle or fleet qualification is inferred. The receipt includes full
model descriptors, exact signed identities, request settings and environment.

The [Reploid physical run](final-physical-reploid.json) also passes generation
and reranking on the same archive. Its complete generation output, including
text, all 16 token IDs, resolved sampling and stopping reason, equals the
standalone result. Reploid embedding is covered by the shared-device run below.

The [physical shared-device rerun](final-physical-shared-device.json) passes:
both sessions submit inference to the same device, closing the first preserves
the second, and complete embedding outputs including per-item receipts match.
The [final full CI](final-ci-green.json) passes 802 test files and browser kernel,
demo controls, contrast and offline checks. Its two initial incomplete fixtures
were repaired without weakening their corruption and cache-integrity assertions.

| Physical operation | Standalone | Reploid |
| --- | --- | --- |
| Qwen3-4B-Instruct-2507 generation | Pass: 16 reference tokens | Pass: identical final output |
| Qwen3-Embedding-0.6B embedding | Pass: two finite 1,024-dimensional vectors | Pass: two sessions share a device; closing one preserves the other |
| Qwen3-Reranker-0.6B reranking | Pass: two-document completion | Pass: two-document completion |

Reproduce the installed checks using the pinned revisions, archive and commands
in [the streaming guide](../../docs/capsule-streaming.md#reproduce-the-focused-evidence).
The physical runner accepts the explicit `config` retained inside each physical
receipt; restore the identified model artifacts and replace local paths for the
new environment. These runs use the recorded retained-local release policy.
They do not authorize release promotion or infer current revocation status.

Earlier archives and failures remain historical. In particular, the generation
device-loss attempt, passing v1 control, missing browser export, package-size
failure and initial shared-session failure do not disappear when a later check passes.

Reploid's full CI separately has 60 failures in seven files, reproduced on
unchanged upstream `2cbe2fc85d8fc4e5a5ba652f962219566c951fed`.
[The paired Reploid record](https://github.com/clocksmith/reploid/blob/codex/consumer-streaming-closure/artifacts/incremental-streaming-2026-09-12/README.md#preserved-failures)
retains that baseline; the installed-consumer pass does not imply full Reploid CI passes.

Component: doppler.runtime-source.client, doppler.runtime-source.gpu,
doppler.tests, doppler.repository-tooling, doppler.docs.
Intent: preserved.
Acceptance evidence: linked installed, transport, lifecycle, physical and CI records.
Boundary effects: explicit public v2 format and helpers; Reploid native journal
layout; preservation of existing shared-device ownership.
