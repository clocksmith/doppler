# Incremental hashing checkpoint

Component: doppler.runtime-source.formats, doppler.runtime-source.storage,
doppler.tests, doppler.repository-tooling. Intent: preserved.
Boundary effects: storage consumes the shared formats hash implementation;
no ownership, retention, numerical, shader, or signed-Capsule changes.

## Reproduction and repair

On the original implementation, a deterministic allocation guard rejected
1,048,640 bytes of padding for an existing 1 MiB input; streaming finalization
requested another 1,048,576 bytes. These are diagnostic observations, not GPU
measurements. The repaired implementation consumes full blocks directly and
retains only hash state, a schedule, and the unfinished block.

`node tests/formats/sha256-incremental.test.js` passes Node crypto comparisons
for empty input, padding boundaries, offset views, arbitrary chunk divisions,
mutated input chunks, repeated finalization, and 537,919,488 streamed bytes.
For a 1,048,635-byte input it measures 512 bytes of one-shot typed-array
workspace allocations and 544 bytes for the streaming byte-digest API; no
individual allocation exceeds 256 bytes. This excludes input ownership,
JavaScript objects, and UTF-8 encoding of string inputs.

`node tests/runtime/capsule-artifact-allocation.test.js` passes deterministic
owned-copy, hashing-workspace, and returned-slice allocation failures, retry,
and close checks. The existing capsule-artifact-retention test preserves
mutation isolation and corruption rejection after eviction. The memory-probe
test uses doubles and does not establish physical GPU behavior.

## Physical gate: hashing-only rerun failed in owned copying

The hashing-only candidate was packed before the incoming main changes:
`/var/tmp/doppler-memory-hash-only-20260920/doppler-gpu-0.6.2.tgz`, SHA-256
`57f086bddb4a53c8b1d658ae8f167dee6c6bdd45715559ad01f1425fb3e292ef`.
Its installed-consumer package tests pass; this is not model qualification.

Run `node tools/check-installed-capabilities.js artifacts/memory-safe-loading-2026-09-20/hash-only-config.json`
with that retained bundle and model volume. The output directory must not
already exist. The config preserves the original generation Capsule, frozen
107-token reference, request, two-session sequence, and unbounded retention.
Only memory observation is added. Runtime source changes in this archive are
limited to SHA-256 and its storage adapter; source revision is 79eed662.

The test completed initial load, two repeated requests, and a request after
cancellation/rejected preparation, all matching the frozen tokens. Opening the
second session failed at `Uint8Array.from` in verified-capsule-artifact-store.js:41,
not SHA padding. Full error, measurements, and observations are in
hash-only-physical-receipt.json. This establishes the need for the next verified
backing-store repair; the original allocation gate remains open. No backing-store
change preceded this result. Memory instrumentation measures WebGPU object
lifetimes, not physical driver residency. Source/copy/hash/retention counters
remain separate; a low hashing workspace does not imply low total retention.

## Main integration

Fast-forwarded to f0ce543f, preserving incoming UI, live-token evidence, and
cache progress changes. Focused incoming generation-evidence and OPFS tests
pass. Generated dependency and runtime-closure inventories are synchronized.
The measured merged package remains 1,858 files (see package-audit.json).
This newer merged source is not the archive in the running physical test.
Full check:green is still running at this checkpoint. No release publication.
