# Authenticated Node opening and backing lifetime

The installed two-model candidate reopened in a median **18.03 seconds**, versus
**51.19 seconds** with JavaScript SHA-256 and the historical secondary digest
loop. These are scoped diagnostics on the existing Linux/AMD machine, with two
fresh processes per arm and warm filesystem caches. Both models stayed resident
for complete embedding-plus-reranking searches. This is not a published release,
browser speed claim, minimum-memory claim, or independent adoption result.

| Installed arm | Median opening | Samples |
| --- | ---: | ---: |
| JavaScript SHA-256, historical secondary digest loop | 51.193 s | 2 |
| Native incremental SHA-256, historical secondary digest loop | 36.493 s | 2 |
| Native incremental SHA-256, optimized compatible digest loop | 18.027 s | 2 |

The arms use persistent disk storage, the same accepted Capsules and reference
corpus, a 1,500,000,000-byte verified-backing budget per model, and zero secondary
cache retention. The first two use the same installed archive with different
loading configuration. The optimized archive also includes a shared-backing
peak-accounting correction and a declaration-only port refinement. Each report
binds its exact runtime archive, application assets, Capsules, hardware, fixture,
and qualification probe. No repository sources were copied into the frozen
consumer after installation.

[opening-summary.json](opening-summary.json) contains individual observations,
medians, and sampled CPU attribution. [evidence-index.json](evidence-index.json)
binds compressed raw reports to their original bytes. Decompress `.json.gz` files
with `gzip -dc` to inspect the original complete reports, including search results
and asset inventories. The archive paths in these receipts are local build
outputs, not published download locations.

## Implemented behavior

- Host hashing uses native incremental SHA-256 in Node and the existing
  incremental JavaScript implementation in browsers. Explicit backend selection
  remains available. Private snapshot blocks, signed digests, and cancellation
  yields are preserved.
- `maxVerifiedBackingBytes` bounds private snapshot leases plus outstanding
  full-artifact reservations. It is distinct from the secondary returned-data
  cache. Shared snapshots count conservatively against each owner's budget.
- Once model preparation and outstanding readers finish, the host releases its
  snapshot leases. A later read reacquires and authenticates bytes unless another
  owner still holds the protected snapshot. No persisted-file trust shortcut was
  introduced.
- Installation now persists the complete artifact closure before recording
  completion. Previously, a model could reuse another live session's bytes and
  omit those files from its own durable installation. Repair of the identical
  accepted Capsule preserves its original acceptance time; it cannot create new
  eligibility after expiry.
- Opening diagnostics distinguish CPU work, buffer allocation and upload calls,
  shader-module creation, pipeline creation, and GPU waits. API durations can
  overlap and are not physical GPU residency measurements.

The baseline CPU profile located the dominant preparation cost in the secondary
digest pass, not shader compilation or layout conversion. The optimized loop
preserves historical digest bytes. Slower intermediate loop variants were
rejected. No shader, precision, tensor layout, model identity, or Run computation
authority changed.

The isolated 64 MiB browser probe measured medians of 512.5 ms for JavaScript and
513 ms for Web Crypto in the tested Chromium. It did not establish a useful
browser improvement or measure Web Crypto's temporary-copy peak. Browser hashing
therefore remains unchanged. Node medians were 523.0 ms JavaScript, 26.9 ms native
incremental crypto, and 44.5 ms Web Crypto. These synthetic probes do not replace
complete-opening qualification.

## Acceptance and memory scope

`final-lifecycle.json.gz` passes unchanged-document reuse, repeated search,
reference-result comparison, indexing and query cancellation after GPU submission,
superseded queries, interrupted saves, explicit closure, corruption repair, and
device-loss recovery. `final-offline.json.gz` passes a fresh process reopening with
kernel-enforced network denial; IPv4 and IPv6 connection probes return `EPERM`.
The application made no network requests. Both use the final installed runtime
SHA-256 `32af41c69fc8f25b04ea0461d32dd8947217bb18a995e68d15fd65b1a643b1c4`.

The optimized comparison processes reported high-water RSS of 1.826 and 1.880 GB.
Verified snapshot peaks were 1,201,886,617 bytes for embedding and 944,962,214 bytes
for reranking, with zero retained snapshot bytes after each load. Requested GPU
buffer allocation peaked at approximately 3.987 GB. These are different measures:
requested allocations are not physical GPU residency, and RSS on this unified
memory machine is not an independently additive GPU-memory figure.

The explicit backing budget does not cover acquisition buffers, returned copies,
loader working data, or GPU resources. Loading still verifies the complete
artifact closure upfront, so this is not yet a shard-at-a-time low-memory loader.
Acquisition and whole-application resource budgets remain separate work.

The original baseline used memory-backed storage; it is retained for numerical
comparison and profiling, not a controlled memory comparison with the disk runs.
The machine remains the 122 GiB Linux/AMD reference configuration. Disk-backed
process restart is established; machine reboot, physical storage exhaustion,
smaller-memory hardware, and ordinary browser configuration are not.

## Compatibility defect and release limit

The pre-existing secondary implementation named `blake3` does not match the
[official BLAKE3 vectors](https://github.com/BLAKE3-team/BLAKE3/blob/master/test_vectors/test_vectors.json).
[legacy-hash-compatibility.json](legacy-hash-compatibility.json) records the empty
input mismatch. Twelve frozen historical vectors and multiple streaming chunk
sizes prove compatibility with retained artifacts, not standard conformance.
Existing immutable manifests require those historical bytes. Correcting that
format requires a distinct qualified artifact release; the Capsule boundary
continues to authenticate artifacts with SHA-256. Use SHA-256 for new artifacts
until a separately qualified correction is available.

The attempted fresh-release build failed with `Release eligibility has expired`
([build.txt](build.txt)). The candidate was then built through the existing
retained-model path and qualified against a previously accepted installation.
Its durable closure was explicitly repaired; no acceptance timestamp was forged
and no signature or checkpoint check was bypassed. Fresh acquisition requires
renewed release eligibility and clean-consumer acceptance. The accepted published
starter archives were not overwritten.

## Repository validation

The full `npm run check:green` sweep passed 863 of 865 test files; its original
output is retained in `green.txt.gz`. The two failures were repaired and passed
through the canonical runner: historical serialized receipt wording changed by
the Forge-to-Rig rename, and the GPU-observer test fixture's incomplete device
implementation. The two serialized historical strings retain their original
bytes; public naming remains Rig and Run. Four focused files pass in
[focused-repairs.txt](focused-repairs.txt).

The final independent gate sweep passes closure, public boundaries, style,
architecture, generated dependencies, browser imports, export parity, source
types, goals, and component inventory. See
[final-gates-repaired.txt](final-gates-repaired.txt). An intermediate stale
dependency inventory failure is retained in `final-gates.txt`; it was regenerated,
not waived. The full 865-file command was not rerun after the focused repairs.
The packed public-consumer and declaration checks pass in
[package-receipt.json](package-receipt.json).

Subsequent compact-candidate work reran the complete repository check successfully:
866 test files and all required gates pass. See the
[later validation receipt](../document-search-compact-2026-09-26/validation.json).
The earlier failed sweep above remains retained as historical evidence.

The audited package adds 6,138 unpacked bytes with no new package files or
dependencies. [package-audit.json](package-audit.json) lists the changes. The
unpacked ceiling was deliberately updated to the exact reviewed payload;
the existing packed ceiling remains unchanged.

Component: `doppler.runtime-source.client`,
`doppler.runtime-source.client.model-host`, `doppler.runtime-source.config`,
`doppler.runtime-source.storage`, `doppler.runtime-source.converter`,
`doppler.repository-tooling`, `doppler.tests`, `doppler.docs`.

Intent: preserved. Acceptance evidence: the installed reports, packed consumer,
focused regression tests, and final gates above. Boundary effects: none; loading
policy and ownership are explicit, application installation stays outside Run,
and Rig → Capsule → Run retains its existing authority split.

The next product work remains a frozen quality corpus and compact qualified pair,
followed by ordinary-machine acceptance, a versioned application recipe, and an
independent integration through a second revision. The 300 MB combined target is
provisional, not an achieved release size.
