# Capsule reopening: JavaScript hashing cost (2026-10-22)

Reopening the issue-triage Capsule (945 MB, 40 artifacts) from OPFS with no network took 13.2 s
in Chrome 146 on a Radeon 8060S ([before](reopen-before.json)). A CPU profile attributed about
7.5 s to the portable SHA-256 that browsers use for Capsule artifact verification
(`src/formats/sha256.js`) and about 2.9 s to BLAKE3 shard verification (`src/storage/blake3.js`).

Both implementations computed 32-bit sums with `>>> 0`, which yields values that V8 cannot keep
as int32, so the inner loops ran on boxed doubles. SHA-256 now keeps state, schedule and round
constants in `Int32Array` and uses `| 0`; BLAKE3 uses `| 0` for its additions and reads full
64-byte blocks four bytes per word. The arithmetic is unchanged modulo 2³², and so are the
private 64 KiB blocks, incremental hashing and yield points documented in the
[root API](../../docs/api/root.md).

| | Before | After |
|---|---|---|
| SHA-256, 64 MiB in Node ([bench](sha256-bench.js)) | 118 MB/s | 267 MB/s |
| BLAKE3, 64 MiB in Node ([bench](blake3-bench.js)) | 305 MB/s | 383 MB/s |
| Offline reopen in Chrome ([script](open-trace.js)) | 13.2 s | 8.5 s, 8.7 s ([1](reopen-after-1.json), [2](reopen-after-2.json)) |

Digests match `node:crypto` for lengths 0–1 MiB across block boundaries and mixed update sizes,
and the official BLAKE3 vectors in `tests/storage/blake3.test.js` pass.

Remaining reopen time: about 3.4 s SHA-256, 1.5 s BLAKE3 over the same bytes, and 3.5 s of GPU
upload and pipeline creation. A WebCrypto artifact backend, or one verification pass instead of
two, would remove most of the hashing, but each changes a documented verification or memory
contract and needs a maintainer decision.

Component: `doppler.runtime-source.formats`, `doppler.runtime-source.storage`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: this directory; `tests/storage/blake3.test.js`, `tests/runtime/capsule-artifact-allocation.test.js`, `tests/runtime/capsule-backing-budget.test.js`, `source:style:check`, `typecheck:source`.
Boundary effects: none.
