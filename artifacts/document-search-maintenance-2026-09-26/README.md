# Document search 0.1.1 maintenance delivery

The browser and Node application archives are published and anonymously verified
at immutable Hugging Face revision `eafe756a6b11d19eafc9e030b720a2c92e63b2c7`.
[Publication identities and URLs](publication.json) bind both archives to the
unchanged candidate runtime SHA-256
`c5be891b3451944f3ff8cf951e46ed6b7fe0f2180d117207bb0b1833669797af`.
The accepted model Capsules, shader bytes, and model shards are unchanged.
The optimization reduces temporary allocations in `src/storage/blake3.js` while
preserving its historical nonstandard digest; it does not replace the algorithm.

## Installed acceptance

Both applications ran embedding and reranking resident together on Linux/AMD
Radeon 8060S. Node used 22.22.1 and `webgpu` 0.4.0; browser used Chrome
146.0.7680.177. Profiles and model state were on persistent ext4 storage.
The host had 122 GiB usable RAM. This is a supported configuration, not a minimum.

- [Public Node consumer](public-consumer-summary.json): downloaded the final
  archive anonymously, verified its hash, installed with frozen dependencies,
  and fetched all 32 model shards into empty storage. Physical reference searches,
  repeated queries, index reuse, submitted cancellation, supersession, interrupted
  saves, closure, corruption repair, and device-loss recovery passed.
- The same public consumer reopened with IPv4/IPv6 socket creation denied by
  the kernel; repeated results, index reuse, and closure passed. The retained
  `public-node-fresh.json.gz` and `public-node-offline.json.gz` bind executable
  assets, runtime, Capsules, lockfile, hardware, corpus, and probe.
- `prior-state-upgrade.json.gz` records that the public maintenance application
  reused a copy of the previous application's documents/index offline and matched
  its recorded searches. The original state was preserved. This is a controlled
  maintenance check, not independent adoption.
- [Browser acceptance](acceptance-summary.json) passed fresh public model
  acquisition, reference searches, the lifecycle cases above, quota failure,
  incompatible-index recovery, and offline reopening with the server stopped.
  The complete receipt is `browser-qualification.json.gz`. The final archive
  changed README/root packaging only; [final executable identity](browser-final-identity.json)
  and [public installed identity](public-browser-identity.json) match the physical
  run, including generated assets. No checkout runtime was copied after install.

Public Node fresh installation measured 112.810 seconds; its subsequent offline
open measured 13.443 seconds. Sampled and process-high-water RSS were
2,221,133,824 bytes during acquisition/lifecycle acceptance. Peak requested GPU
buffer allocation was 3,989,755,736 bytes; this is not physical GPU residency.
Browser installation measured 127.455 seconds and offline reopening 28.591
seconds. These acceptance observations are distinct from the balanced study.

## Balanced Node opening comparison

The [frozen protocol](opening-study.json) ran three separately launched blocks
(ABBA, BAAB, ABBA), with a fresh Node process for each of 12 observations.
There were six runs per package, on the same boot, disk state, and hardware,
with warm operating-system filesystem cache. All runs passed all 12 searches,
unchanged-index reuse, and closure. No samples were dropped.

| Package | Median opening |
| --- | ---: |
| Previous runtime `32af41c6…` | 18,060.371 ms |
| Retained candidate `c5be891b…` | 12,455.374 ms |

The [full observations and summary](opening-summary.json) support a local
31.035% reduction. This is not a cold-filesystem, reboot, smaller-machine, or
paired browser performance claim. Individual policies and compressed receipts
are retained here.

## Eligibility and evidence custody

New signed eligibility events extend the existing histories, retaining their
checkpoints and previous events. Expiry is enforced; explicit renewal requires
publisher signing custody and cannot erase a denied history. Existing expiry
policy is unchanged. **Fresh installation from these static archives requires
eligible metadata before 2026-09-28 00:55:40.930 UTC (browser) or
2026-09-28 00:59:12.562 UTC (Node).** Later fresh installations require renewed
signed metadata in a new deliverable. An already accepted installation uses its
explicit retained-local authorization for offline reopening; unseen revocations
cannot be discovered offline. [Exact per-model expirations](fresh-install-eligibility.json).

Both complete F16/Q4 numerical captures were copied from `/dev/shm` to disk,
compressed, published immutably, and verified after anonymous download and
decompression. [Archive hashes](full-diagnostics-archive.json) and
[public verification](publication-verification.json) retain this custody. Original
captures remain untouched. Byte-identical immutable historical shard copies were
consolidated using hard links after comparison; paths and bytes were retained.
The storage audit is `storage-deduplication.jsonl`.

## Scope and next work

Machine reboot, actual disk exhaustion, smaller-memory hardware, browser peak
process memory, and independently maintained adoption remain unestablished.
Quota failure was injected; it does not establish physical disk-full behavior.
The archive supplies ranking, with no qualified answerability/abstention policy.
Qwen precision and abstention rejections remain unchanged. MiniLM source intake
is [separate unpublished work](../document-search-minilm-embedding-2026-09-26/README.md).

[Branch inventory](branch-inventory.json): all 19 local and 16 GitHub branch tips
were ancestors of `main`; fetch explicitly included every remote head despite
the configured main-only refspec. Uncommitted work in the separate local-journeys
worktree remains intact and is not represented as merged commits.
A subsequent fetch found `8f4742bd` on GitHub `main`; it was fast-forwarded without
altering local changes. The [final inventory](branch-inventory-final.json) again
shows every local/remote tip contained in `main`.

Repository validation ran all 868 test files successfully. The aggregate found
two failures caused by the generated runtime closure's stale ModelIR hash.
Regenerating that receipt changed one source record, preserved the 71-file
closure, and both `runtime:closure:check` and installed `package:smoke` passed on
rerun. All required gates have passing evidence across the aggregate and these
targeted reruns; the failed aggregate and repair logs are retained explicitly.

Component: `doppler`, `doppler.repository-tooling`, document-search application.
Intent: preserved.
Acceptance evidence: installed receipts and balanced study above; repository
validation is recorded in `validation.json`.
Boundary effects: explicit publisher-owned eligibility renewal and browser
archive assembly; consumer expiry/trust checks and Run computation are unchanged.
