# Capsule GPU token selection — retained local acceptance

Implementation: `eaea457ef793f5846f8603bf35af832e54a0e4f0`, stacked on the
registry/interface repair in PR #11. [PR #12](https://github.com/clocksmith/doppler/pull/12)
adds an explicit `doppler.capsule-token-selection/v1` TargetPlan declaration.
Declared generation keeps the finalized f32 scores, penalties, suppression,
and token selection on the GPU. One selected u32 token is copied in the same
submission as selection. Existing Capsules retain their declared CPU control
path. Full-score diagnostic methods remain available.

**Result:** the installed standalone and Reploid paths reproduce the retained
tokens, verify incremental completion, and preserve stopping and cleanup.
Readback volume falls substantially. These runs do **not** show an end-to-end
latency improvement; this is not a performance promotion or publication.

## Exact installed identities

| Item | Identity |
| --- | --- |
| Local Doppler candidate | `doppler-gpu@0.6.2`, SHA-256 `9c574a03500e137e44184cc1827339ca4b4ee6e60597351999d5f9d0679c06af` |
| Local candidate size | 2,117,296 packed bytes; 11,118,200 unpacked bytes; 1,804 files |
| Installed Reploid library | SHA-256 `b92d6ac8a88740aebe82745d7b2e67afd9eb6c8f676efc9bed675f1cad35a250` |
| Local Reploid fixture revision | `19bc359ef1b7606fde3132ab2f37920f4891e6e5` |
| Baseline Doppler archive | SHA-256 `f9758530d0fec3967dfa96dc64252dcdd874c0c73f120ee28f7767899f0ca2d6` |
| Physical surface | Chrome 146.0.7680.177, Linux, AMD Radeon 8060S / RDNA 3, non-fallback adapter |
| Local pack toolchain | Node 22.22.1, npm 9.2.0, zlib 1.3.1 |

[Package reproduction](package-reproduction.json) establishes that packing the
committed implementation with the same local toolchain reproduces the tested
archive exactly. The generated inventory correction changes no package bytes.
CI builds a separate archive and shares those bytes across its own standalone
and Reploid consumers. [CI observations](ci-observation.json) retain its workflow
and artifact identity; that artifact ZIP digest is not the local runtime digest.
Published 0.6.1 remains a separate compatibility control.

## Physical generation and transfer observations

The Qwen3-4B f16-weight/f32-activation evaluation Capsule contains a newly
declared execution recipe. Its semantic root is
`sha256:081a4e03adbdc572d778cb9ba4470989bb5d5b2e9e638c49db2a390460449516`;
its TargetPlan is
`sha256:2983e5f3b9a9c2e03825ba0168c5c8272788a30e1879fcfd67d0836ed5b5b496`.
See [the Capsule](evaluation-capsule.json), [qualification](qualification.json),
[Forge receipt](forge-receipt.json), and public-key-only [open options](open-options.json).

All three final standalone requests matched the baseline's 256 token IDs and
stopped at the output limit. Reploid matched all 107 source-reference tokens,
including EOS, emitted 106 partials, and displayed the verified final text.
The source model revision is `cdbee75f17c01a7cc42f958dc650907174af0554` of
`Qwen/Qwen3-4B-Instruct-2507`. The old source transcript is a retained reference;
the new installed runs execute the actual model. This does not qualify arbitrary
prompts, stochastic model outputs, adapters, other models, or other hardware.

| Measurement | CPU-selection baseline | Final GPU-selection candidate |
| --- | ---: | ---: |
| Vocabulary | 151,936 | 151,936 |
| Score/token readback per decode | 607,744 bytes | 4 bytes |
| Observed decode readback, 255 steps | 154,974,720 bytes | 1,020 bytes |
| Observed decode submissions, 255 steps | 1,530 | 1,530 |
| Observed decode dispatches, 255 steps | 111,180 | 111,690 |
| Unobserved repeated request, 256 tokens | 17,197.9 ms | 17,523.2 ms |
| Unobserved repeated median decode interval | 63.0 ms | 64.6 ms |

These are local diagnostic observations, not a randomized benchmark population
or a vendor comparison. The candidate's last unobserved request was about 1.9%
slower. API copy sizes do not measure physical bus throughput. Queue and map
waits overlap and must not be summed. CPU profile samples are estimates, not
independent additive timers. Reused buffer labels in the trace describe pool
history, not the current tensor's semantic role.

Full records: [baseline](baseline-receipt.json),
[final standalone](standalone-receipt.json),
[final analysis](gpu-selection-public-summary-final.json), and
[physical Reploid](reploid-physical-receipt.json).

## Correctness and lifecycle coverage

- Physical operator regressions: 129 Capsule selection cases, 540 sampling
  distribution cases, and 96 history-penalty cases. The selection checks include
  ties, suppression, invalid scores, and empty candidate sets. The independent
  non-finite probe covers 12 additional input/output observations.
- Failure injection: abort and device replacement during compilation, abort
  before submission, post-submission cancellation, mapping failure, and cleanup
  of operation-owned logits/hidden buffers. Submitted GPU commands are not
  promised to be interrupted.
- Installed fixtures: both stream formats, final reconstruction/identity,
  cancellation and reuse, request-bound adapter lifecycle, owned versus borrowed
  sessions, and rejection of a score-readback attempt on the declared token path.
  These fixtures use injected programs and are distinct from physical execution.
- Registry/interface generation, source types, source architecture/style,
  public boundaries, and package declarations have retained passing checks.

The first complete local unit invocation passed 822 of 825 files. Its three
failures depended on retired shader identities in test fixtures. The repaired
tests preserve the historical receipts and independently bind current candidate
source bytes; no historical model manifest or qualification was repinned.
[Unit results](unit-results.json) retain this initial failure and focused reruns.
The [full local acceptance-chain rerun](local-acceptance.json) passed, including
all 825 unit test files and installed-package contracts.
The [remote acceptance chain](ci-final-acceptance.json) also passed at
`a99c7994`, including all 825 unit files, browser WebGPU (71 passed, 15 skipped), demo UI/offline checks,
and the separate installed-consumer workflow. Subsequent retained-evidence
changes do not alter the accepted runtime archive.

## Reproduce within the recorded local environment

The archives, weights, and full CPU profiles remain at the paths in the configs.
They are not embedded in the runtime npm package or this evidence directory.

```sh
TMPDIR=/dev/shm/doppler-runtime-coherence node tools/profile-installed-capsule.js artifacts/runtime-coherence/gpu-token-selection/gpu-selection-public-config-final.json
TMPDIR=/dev/shm/doppler-runtime-coherence node tools/check-installed-capabilities.js artifacts/runtime-coherence/gpu-token-selection/gpu-selection-reploid-physical-config-final.json
```

The installed archive and model identities must remain exact.
[Migration instructions](../../../docs/integration/capsule-token-selection.md)
describe adoption through existing `openCapsule()` sessions. Forge requires
observed initial execution identity and exact shader closure before sealing.

The preliminary probe and first candidate are retained: the first probe was
interrupted after an inefficient harness read pattern; the first public candidate
used an extra selection readback submission. The final candidate co-records that
copy. Non-finite selection and failure-path resource leaks were reproduced and
fixed. Failed attempts remain evidence, not accepted runtime identities.

Component: `doppler`. Intent: preserved. Acceptance evidence: linked installed,
physical, regression, and CI records. Boundary effects: declared Capsule
generation, Forge recipe validation, GPU sampling and resource cleanup, and
installed-consumer test fixtures. No Reploid implementation change, deployment,
npm publication, production signing authority, or independent adoption is claimed.
