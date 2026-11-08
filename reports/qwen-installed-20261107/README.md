# Installed Qwen artifact repair and bounded execution checks

These are functional checks on an AMD Radeon 8060S (WebGPU/Dawn, Mesa
26.0.3), alongside an unchanged running 27B provider. They establish neither
isolated throughput nor full-model numerical/quality qualification. The
Flash-Next long-context gate remains open; no precision or tolerance policy
changed here.

## Artifact boundary

Several historical manifests used obsolete shard digests. The migration
independently authenticates manifest/tokenizer Git blobs and every shard against
an immutable Hugging Face Git-LFS SHA-256 inventory before rebuilding standard
SHA-256 metadata. Runtime verification stays strict; no retry bypass or hash
fallback was added. Weight and tokenizer bytes are unchanged.

- [Small-model publication](publication-small.json): Qwen 3.5 2B, Qwen3
  Embedding 0.6B, and both Qwen3 Reranker 0.6B variants. The 0.8B repair and real
  browser OPFS evidence live in [the earlier report](../qwen08-cache-repair-20261107/README.md)
  and [activation report](../demo-activation-20261107/README.md).
- [27B publication](publication-27b.json): Qwen 3.6 27B F32 and F16 execution
  variants share authenticated Q4K weights. A source-aware converter refresh
  supplies missing modern inference-contract fields. The F16 variant references
  the refreshed F32 capsule and exact manifest digest, not its retired identity.
- [Pinned source config](qwen36-source-config-pin.json) governs the refreshed
  contract. Original weight conversion records still say `sourceRevision: main`;
  pinning a config now does **not** retroactively pin the original source weights.
  Converted bytes are independently pinned by the retained Git-LFS inventory.
- Preparation can return authenticated metadata in memory for converter repair.
  The migration writer still refuses an incomplete runtime contract before
  creating any artifact. [Regression log](migration-check.log).

## F16 dispatch failure and repair

The F16 vocabulary projection attempted 248,320 X workgroups, exceeding WebGPU's
65,535-per-axis limit. [Original failure](smoke-27b-f16.json) and
[CPU regression](dispatch-before.log) are retained. The declared dispatch now
spreads columns across X/Z while Y remains the independent row-tile axis. Its
explicit uniform layout includes the X width. Arithmetic and accumulation are
unchanged.

[Metal](wide-grid-metal.json) and [AMD](wide-grid-amd.json) each check 576,134
F16 outputs against an exactly representable independent rational fixture,
including a 248,320-column head, partial column/row planes, and output guards.
Every output bit matches. The formerly failing full F16 model now produces
`Hello` on both repeated greedy requests: [fixed smoke](smoke-27b-f16-fixed.json).
This is a geometry repair, not evidence that all-F16 is accurate for every task.

## Exercised artifacts

| Artifact | Profile | Retained result |
| --- | --- | --- |
| Qwen 3.5 0.8B Q4K | `profiles/qwen-3-5-0-8b-vulkan-throughput` | [Repeated greedy generation](smoke-08.json) |
| Qwen 3.5 2B Q4K | `profiles/qwen-3-5-2b-vulkan-parity` | [Repeated greedy generation](smoke-qwen-3-5-2b-q4k-ehaf16.json) |
| Qwen3 Embedding 0.6B Q4K | manifest/default | [Finite 1,024-dimensional embedding](smoke-qwen-3-embedding-0-6b-q4k-ehf16-af32.json) |
| Qwen3 Reranker 0.6B Q4K | manifest/default | [Finite scores, relevant document ranked first](smoke-qwen-3-reranker-0-6b-q4k-ehf16-af32.json) |
| Qwen3 Reranker 0.6B F16 | manifest/default | [Finite scores, relevant document ranked first](smoke-qwen-3-reranker-0-6b-f16-af32.json) |
| Qwen 3.6 27B Q4K/F32 | manifest/default, 24 GiB test pool cap | [Repeated greedy generation](smoke-27b.json) |
| Qwen 3.6 27B Q4K/F16 | manifest/default, 24 GiB test pool cap | [Repeated greedy generation after dispatch repair](smoke-27b-f16-fixed.json) |

The Q4K reranker declares `true_logit`; its F16 sibling declares
`logit_difference`. This repair preserves those distinct existing scoring
contracts. Selected 0.8B/2B profiles remain opt-in; no global fastest-path claim
or default precision switch follows from these smokes.

Reproduce with
`DOPPLER_NODE_WEBGPU_MODULE=webgpu node tools/experiments/qwen-installed-smoke.js MODEL DIRECTORY PROFILE OUTPUT --physical`.
The retained runs used the same operations/assertions in an operator wrapper
whose absolute paths are now CLI arguments. GPU evidence uses the archived
`46aa1df32f2ff6278d01cbf1c4022a481c360f89` tree plus the wide-grid runtime delta
retained with this change. Source provenance is recorded in `provenance.json`.

## Consumer and evidence limits

Reploid's identities differ from the refreshed Doppler manifest identities.
The initial `reploid-smoke-08.json`/`reploid-smoke-2b.json` exercise controlled
replacement artifacts, not the public catalog. Exact-catalog follow-up receipts
are separately named `reploid-catalog-*.json`; they also test pinned piece
acquisition and approved-directory confinement before actual GPU execution.
No participant was announced, replaced or restarted. The 35B check uses its
existing Q4_0 artifact, not Q4K. These are offline native-provider checks, not
remote browser/mesh or production-signing qualification.

Public JSON copies normalize private operator paths; they are not original
signed execution receipts. [Evidence integrity](evidence-integrity.json) binds
each original and published byte representation. Config/inventory files are
retained verbatim where their exact content is part of provenance. Older
failed discovery/metadata attempts remain in `results.json`; later successful
smokes do not erase them. No model source was deleted. Native-directory
confinement required owned copies for two small Reploid artifacts because the
models volume cannot hardlink/reflink; Doppler views reference existing shards.

## Installed and deployed result

The committed `019dec78` runtime archive was installed independently of the
in-progress snapshot and passed [actual generation/reuse](final-install-smoke.json)
using the permanent authenticated 0.8B artifact view. The operator's private plan
records exact runtime/artifact paths, selected profiles and the borrowed Node
provider dependency; no daemon was started. Native Reploid checks on all four
catalog identities passed: [0.8B](reploid-catalog-08.json),
[2B](reploid-catalog-2b.json), [27B](reploid-catalog-27b.json) and
[35B](reploid-catalog-35b.json). The consumer fixture and original receipts are
also committed in Reploid `6f4ceecfe`.

[The deployed release](deployed-release.json) pins source `019dec78`, verified
0.8B/2B quickstart URLs, and the hosted shell inventory. Local-preview and
production browser control checks passed with mocked generation; physical
production capture is recorded [separately](hosted-physical-final.json): real
OPFS import, `Hello!`, 72/72 prefill/decode frames, zero gaps and browser errors.
[Grid screenshot](hosted-grid-final.png). No npm publication or replacement
of the running provider is implied by this web deployment.

Component: `doppler.repository-tooling`, `doppler.tests`, `doppler.runtime-source.gpu.kernels`
and artifact catalog. Intent: preserved. Acceptance: retained byte-authentication,
physical Metal/AMD geometry results, per-model AMD smokes, focused migration and
dispatch tests, source typing/architecture/unused checks, kernel generators,
runtime/package closure, quickstart/shell checks, and hosted browser controls.
Boundary effects: D4DA hosts the explicitly pinned runtime; Reploid received
only a test fixture and evidence, with no product/bundle or identity change.
