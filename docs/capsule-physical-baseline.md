# Capsule inference and installed document search

The local physical baseline now covers signed Capsule inference in Node and
Chromium, actual embedding plus reranking, persistent browser installation, and
resumable model onboarding. The machine is an AMD Radeon 8060S / RDNA 3 with RADV
STRIX_HALO. Both execution surfaces rejected software fallback adapters.

The machine-readable [acceptance receipt](../reports/capsule-baseline/20260906/acceptance.json)
pins the reports, archive, source snapshot, and custody inventory. Detailed data
lives under `/home/x/.local/share/doppler/capsule-baselines/20260906`; the working
experiment directory is `reports/capsule-baseline/20260906`. This is local custody,
not a remote backup or publication. Private evaluation signing keys remain in
custody directories and are excluded from the installed browser application.

The subsequent `rdpull` integrated upstream `b3004aa1`; its
[integration receipt](../reports/capsule-baseline/20260907-rdpull-b3004aa1/acceptance.json)
records validation of the combined tree separately from these physical results.
Upstream also retains a [0.6.0 publication receipt](../reports/release-qualification/0.6.0-20260907/publication/receipt.json)
for a different archive. The physical reports below remain bound to their original
archive and source snapshot; pulling newer source does not requalify them.

## Results and limits

- The latest tested `doppler-gpu` archive is version 0.6.0, SHA-256
  `9bc30b7d39cfee870ffd26f6e0f6011e3f77adaed6a51eb66a27408c8a7aa81f`.
  Installed Node and browser code perform real signed inference. Package smoke
  evidence separately checks 28 exports and declarations with synthetic execution.
- The reranker retains source revision
  `e61197ed45024b0ed8a2d74b80b4d909f1255473` and its original oracle and tolerances.
  Node and browser recovery cover physical device loss and successful reopening.
  Cancellation, interrupted loading, cleanup, corrupted artifacts, retained use,
  known revocation, and rollback have separate receipts. The original revoked
  checkpoint remains at sequence 2 with its original digest.
- Embedding onboarding completes for
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and
  `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418`. Each run converts 310 tensors,
  independently captures 18 reference vectors, compares physical model outputs,
  constructs a signed Capsule from source ModelIR, and qualifies installed
  Capsule execution. Frozen token policy and numerical tolerances stay unchanged.
  Only README bytes differ between these source revisions; this exercises a
  related source revision, not new weights or a model-quality improvement.
- The text/Markdown search fixture contains six unseen documents and six queries.
  Actual models meet all six top-result expectations; BM25 meets five. Those five
  incumbent victories remain in every comparison. These results qualify this
  fixture, not general retrieval superiority or external adoption.
- Closing Chromium, stopping the application server, disabling networking, and
  reopening the same browser profile preserves search. An explicit application
  update to a newly qualified embedding Capsule rejects the old vector index,
  rebuilds from retained documents, preserves prior release checkpoints, and
  passes offline reopening. All six updated online/offline scores match exactly.
  Browser-context observations, including service workers, record zero requests
  during online reembedding and search and no page WebSockets.
- Interrupted installation, physical OPFS corruption and repair, Chromium origin
  quota exhaustion, simulated index incompatibility, and physical GPU device loss
  are retained separately. The actual application-update receipt supplements the
  simulated incompatible-binding check.

## Measured retention tradeoff

The controlled comparison alternates six fresh Chromium processes, three per
policy, on the same hardware, installed archive, signed models, retained storage,
and frozen query set. Operating-system caches are not claimed cold.

| Median observation | Unlimited retention | 128 MiB per model |
| --- | ---: | ---: |
| Renderer peak RSS | 3,024,470,016 bytes | 1,127,718,912 bytes |
| Retained verified artifacts after opening | 2,145,446,153 bytes | 250,826,752 bytes |
| Opening | 64,921 ms | 184,892 ms |
| First query | 1,182 ms | 1,179 ms |

The benefit is lower renderer memory; evicted files increase reads and hashing.
The default stays unlimited. RSS includes shared mappings and excludes the
browser GPU process; it is not unique system memory or total GPU residency.
Artifact counters separately record acquisition, hashing, copying, and retained
bytes. Native upload submission counts and queue completion waits are recorded
without claiming isolated transfer throughput. Every comparison preserves full
rankings and the frozen score tolerance. No numerical kernel changed.

## Reproduction

Retain the custody inventory and source snapshot together. The snapshot includes
repository tooling and configuration; installed archives, source checkpoints,
reference outputs, application builds, browser profiles, and attempt receipts
are separate retained files. Source references record Python, Torch, and
Transformers versions; browser reports record the adapter and browser version.
Several historical configurations contain absolute workspace paths. Restore those
paths, or author new pinned configurations and use new output directories. Do not
edit signed artifacts or retained configuration snapshots in place.

Use the [application instructions](../examples/document-search/README.md) to build
and qualify installations. `application-update-config.json` identifies the tested
update and `retention-comparison-config-02.json` identifies the controlled memory
comparison. Qualifiers create new output directories; preserve completed runs.

The [onboarding playbook](developer-guides/model-onboarding-playbook.md#resumable-embedding-execution)
describes the execution contract. The retained `onboarding-inputs-97b0c614-02` and
`onboarding-inputs-c54f2e6e` directories contain the pinned source specification,
vocabulary, recipe, reference policy, and execution plan. Completed runs can be
revalidated with:

```sh
TMPDIR=/var/tmp node tools/forge-source-truth-model-ir-v2.js \
  --config reports/capsule-baseline/20260906/onboarding-inputs-c54f2e6e/onboarding.json \
  --execution reports/capsule-baseline/20260906/onboarding-inputs-c54f2e6e/execution.json \
  --out reports/capsule-baseline/20260906/onboarding-c54f2e6e
```

The first workflow also retains an injected interruption after real conversion,
its resumed reference attempt, unchanged conversion bytes and modification time,
and successful revalidation without stage execution. Both CLI replay and the
stage-controller regressions reject changed inputs or completed outputs.

Do not replay the original recovery script against its now-revoked checkpoint.
The final Node probe explicitly checks that the old denial and rollback still
reject while a separately qualified source-ModelIR candidate executes. Its new
application binding appends an eligible event; the preceding event and old denial
remain retained. The browser-only plan's rejection on Node is expected surface
qualification evidence. Failed preparation, installation, and pre-fix device
recovery reports remain available alongside successful replacements.

`TMPDIR=/var/tmp npm run check:green` passes all 778 test files, including the
36 Capsule contract files, plus the required architecture, schema, declarations,
package, policy, and charter checks. These are local results; no commit, push,
npm publication, adjacent-repository update, or voluntary external retention is
claimed by this evidence.

Component: doppler. Intent: changed; compiler/runtime ownership preserved.
Acceptance evidence: the pinned physical reports, controlled comparison, installed
archive smoke receipt, onboarding resumes, and complete local acceptance.
Boundary effects: application storage and index ownership, optional runtime file
retention, GPU loss recovery, and resumable development orchestration. Application
release acceptance remains explicit; onboarding cannot publish or approve upgrades.

## Subsequent committed release reproduction

The [20260907 release acceptance](../reports/capsule-baseline/20260907-release/acceptance.json)
records a separate qualification of archive `61a23b9cafd17eb2…`, with tooling and
remote CI bound to commit `984d25633566420e6791c57ce271e388c4bc05b5`.
The isolated checkout passes 779 test files. Public-source reconstruction verifies
126 signed artifacts, and reconstructed models pass installed Node and Chromium
inference, offline search, and recovery on AMD Radeon 8060S. Historical evidence
above retains its original source and archive identities.

The [reproduction instructions](capsule-release-reproduction.md) use the committed
archive, public dependency lockfile, signed metadata, and pinned upstream sources.
The evidence bundle includes full acceptance logs, failed attempts, successful
restorations, and physical reports. This archive is publicly retained in Git;
it is distinct from the npm registry's `0.6.0` archive. Other devices, Bun,
generation, and external adoption remain separate qualification work.
