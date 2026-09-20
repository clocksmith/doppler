# Memory-safe loading acceptance

Final physical status: the original unbounded-retention two-session case and the
separate 4 GiB byte-cache control both pass through installed public APIs on the
retained archive below. Each produces four exact matches to the frozen 107-token
reference. The historical checkpoints explain why hashing alone was insufficient
and why the subsequent backing-store change was required. This is scoped Qwen3-4B
generation acceptance on Linux/Chrome 146.0.7680.177 with an AMD RDNA-3 adapter,
not universal model, hardware, Bun, embedding, or adapter qualification.

## Hashing-only checkpoint

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
backing-store repair; the original allocation gate remained open at this checkpoint. No backing-store
change preceded this result. Memory instrumentation measures WebGPU object
lifetimes, not physical driver residency. Source/copy/hash/retention counters
remain separate; a low hashing workspace does not imply low total retention.

## Main integration

Fast-forwarded to f0ce543f, preserving incoming UI, live-token evidence, and
cache progress changes. Focused incoming generation-evidence and OPFS tests
pass. Generated dependency and runtime-closure inventories are synchronized.
The measured merged package remains 1,858 files (see package-audit.json).
This newer merged source was not the hashing-only physical archive.
Full check:green was still running at this checkpoint. No release publication.

## Measurement-led backing repair (after checkpoint)

The first backing experiment used immutable Blobs, but the installed physical
test failed before load with NotReadableError. An isolated 512 MiB Blob failed
with the same error; chrome://blob-internals reported ERR_OUT_OF_MEMORY. The
Blob design is rejected, not shipped as a successful repair. See
blob-rejected-summary.json and the retained backing-config.json. Chromium's
[Blob storage design](https://chromium.googlesource.com/chromium/src/+/HEAD/storage/browser/blob/README.md)
describes its separate memory/disk quotas; available host RAM alone does not
establish that this representation works.

The replacement owns private 64 KiB byte blocks and shares only completed,
verified snapshots among live stores belonging to one explicit host-service
owner. Public APIs cannot obtain or mutate those blocks. Each store separately
validates its signed closure and permissions. Pending acquisitions and their
cancellation remain independent. The final lease removes shared backing; no
unrestricted global cache or durable disk dependency is introduced.

The optional byte cache accelerates full metadata reads; weight ranges use
owned backing directly. Its limit remains unchanged (including null/unbounded
in the original reproduction), but cache eviction no longer triggers source
acquisition or rehashing. Backing is model-sized memory, separately measured,
not a claim that a 4 GiB cache bounds total model memory.

Focused retention/allocation tests cover shared leases, out-of-order and repeated
close, source/caller mutation, fresh-store corruption, cancellation isolation,
fixed snapshot-block allocation, and failed-read retries. The installed package
test caught a missing declaration in the initial block-backed archive. The
generated package closure now includes that declaration (one additional entry,
no new runtime module). The corrected installed-package test passes.

The corrected candidate is retained at
/var/tmp/doppler-memory-blocks-v2-20260920; blocks-config.json runs the original
physical case without changing browser, model, references, or retention settings.
The original physical case now passes (blocks-physical-summary.json): all four
completed requests match the frozen 107-token reference, including second-session
execution after first-session close. Cancellation/rejected preparation checks pass.
The first session reads and hashes 8,050,837,118 artifact bytes; the second reads,
hashes, and copies zero additional backing bytes. Each close reports zero remaining
backing leases and byte-cache retention. The runtime archive SHA-256 is
`b2ba7dbf7a18df458a702881439d2f9195b8b2a6a864f5fed2fa51afba831f6c`.

The second `npm run check:green` exits zero with 850 unit test files passing.
Focused allocation tests and the CI-toolchain package budget also pass. Incoming
UI defaults are preserved; two stale tests now expect X-Ray/perplexity to start
disabled. The separate 4 GiB retention control also passes against the same archive
(blocks-4gib-config.json and blocks-4gib-physical-summary.json). It completes all
four reference requests with zero cache evictions. Across both sessions the store
reads and hashes 8,050,837,118 source bytes once, rather than reacquiring artifacts
after eviction. The second session adds zero source reads, hashed bytes, or backing
copies; returned slices and GPU materialization remain separately measured work.
The original run takes 490,552 ms and the control 478,945 ms; these are acceptance
durations, not a controlled performance comparison.

GPU observation at final close reports 4,949,428,824 bytes created but not explicitly
destroyed (486 objects), separately from backing leases. Device-pool lifetime,
garbage collection, and physical driver residency are not equivalent to session
ownership; this is not a claim that close immediately returns all process/GPU RAM.
The browser context subsequently closes without a cleanup error. No GPU pooling
policy or completed session-exclusion mechanism was changed.

Component: doppler.runtime-source.client, doppler.docs, doppler.tests,
doppler.repository-tooling. Intent: preserved. Boundary effects: host supplies
an opaque backing owner to Capsule execution; permissions and session state are
not shared. API documentation and generated package/runtime inventories track
the changed storage lifecycle and observation fields.

## Final integration acceptance

The installed-consumer workflow for runtime commit `00d6e6ee` passes:
[GitHub Actions run](https://github.com/clocksmith/doppler/actions/runs/35485721486).
The final browser WebGPU suite passes in its declared SwiftShader lane with
unsupported-capability skips; that suite does not replace the physical model runs.

The real-page demo contract exposed a stale expectation of the incoming UI defaults
and a notice that ignored the enabled Tokens observer. The notice now describes
the actual guided-quality policy, including after loading the sample inspection.
The incoming UI, defaults, and execution policy are preserved. The real-page
`npm run test:demo:contract` passes with mocked model execution and no fatal console
errors. This is UI contract evidence, not another physical generation run.
The final shell digest is
`sha256:bba14ee9eee6e1a5c753c13ed3e6c427833922a78fe28b066de28e83f69379a1`.

The `doppler-debug` protocol kept the repair measurement-led: reproduce after
hashing alone, then change backing ownership only after the original case still
failed. No numerical kernels or frozen references changed. No release is published.

Component: doppler.demo, doppler.tests. Intent: preserved.
Acceptance evidence: `npm run test:demo:contract`,
`npm run demo:reachability:check`, `npm run typecheck:source`,
`npm run catscan:check`, and `checks.json`.
Boundary effects: none; notice presentation follows existing observation policy.
