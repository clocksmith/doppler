# Bounded consolidation: loading acceptance

This is the first independently reviewable change in the requested consolidation,
not a claim that GPU attribution, operation contracts, cleanup, or model work is
finished. No publication or release promotion is performed.

## Reproduced problem and repair

At unchanged `1ab605f2`, a timer-triggered cancellation of a 16 MiB verification
could not run before the entire synchronous copy/hash completed: 309.071 ms total,
297.975 ms hashing, all 16,777,216 bytes admitted. The new regression repeats that
failure mode with 2 MiB for both readers and requires rejection
before complete verification and without publishing backing.

Capable sources now implement optional `streamArtifact(artifact, options)`;
`readArtifact()` remains supported. HTTP uses a bounded BYOB reader when available;
Node files reuse one bounded read buffer. The private snapshot still owns 64 KiB
blocks. Only those owned bytes are hashed, and complete size/digest verification
precedes shared publication. Verification yields through timer tasks, not merely
microtasks. Transport remains outside `formats/sha256.js`, which is unchanged.

The default source-buffer bound is 1 MiB per active streamed acquisition. This is
not a bound on total verified backing, simultaneous acquisitions, browser network
internals, returned slices, or model/GPU memory. Legacy whole-artifact sources
still allocate their complete response. Peer hosts may implement the same port;
no peer orchestration enters inference or hashing.

Tests cover reused/mutated source buffers, arbitrary boundaries, corruption,
truncation, oversize/backing-view attacks, source failure/retry, timer cancellation,
close before publication, and independent overlapping session cancellation.
Artifact-reader declarations are canonical; HTTP and Node implementations join
the checked-JavaScript roots (9 roots, 1,047 unchecked modules).

## Frozen packages and physical scope

- Unchanged baseline: `c3166b1a6a480f52dd0661b67b427f6918eb004b015da84d31bbc86ccb57f13c`.
- Rejected first candidate: `9e9564a42fa87dc34820dedb02c843f09d50aa5222d92f3abb1ca6b33ecf3be4`.
  Real Chrome returned a default-reader buffer larger than the declared bound.
  Its failure receipt is retained; the bound was not increased to hide it.
- Corrected candidate: `58a38aa3eed85a62682ec0a6c2232217c896a7a76393335f6f3a73d46f560de1`,
  `/var/tmp/doppler-consolidation-stream-v2-20260920/doppler-gpu-0.6.2.tgz`.

The fresh baseline and corrected candidate use identical generation Capsule,
configuration, and frozen 107-token reference on Chrome/AMD RDNA 3. Both complete
the four-request, two-session lifecycle. Complete experiment elapsed times are
231,405 and 227,220 ms respectively; these are not single-request startup or TTFT.
Initial hashing is 120,707.5 versus 120,677.1 ms: no hash-speed improvement is
established. Candidate acquisition reports a 1,048,576-byte peak source chunk,
65,536-byte peak backing block, and 7,677 verification yields. Second-session
source reads, hashing, and new backing remain zero.

`hash-throughput.json` independently measures hashing 16 MiB of byte value 19
and checks against Node `createHash('sha256')`: one warmup and three timed runs.
Other checks were active, so it is not a controlled optimization comparison.
Reproduce by calling `sha256BytesHex(input)` four times around `performance.now()`;
allocate the input and calculate the independent reference before timing.

Run physical cases with:

```sh
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-baseline-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-4gib-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-embedding-reranking-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-adapter-config.json
node tools/check-installed-capabilities.js artifacts/bounded-consolidation-2026-09-20/loading-bounded-reploid-v1-config.json
```

Each summary binds its original raw receipt hash and exact archive. Raw receipts
and archives remain at their recorded local paths. Existing numerical references
are unchanged; zero-delta adapter acceptance is not learned-adapter qualification.

The first Reploid run reused a v2 request fixture, but current Reploid main
`67a64fb16661eaa0a8ef34314f3fea824af5be94` declares v1 requests. Its rejection is
retained in `loading-reploid-request-mismatch-summary.json`. The corrected config
explicitly uses that supported v1 public contract, without changing model input,
reference, package, or Reploid code.

The separate `hash-state-read-*-probe` experiments compare just typed-array
destructuring with eight direct state reads. Both agree with Node's independent
SHA-256. These are diagnostic probes, not changes in the loading candidate above.
Reproduce from the repository root, passing the retained baseline implementation:

```sh
node artifacts/bounded-consolidation-2026-09-20/hash-state-read-probe.js /var/tmp/doppler-consolidation-stream-v2-20260920/consumer/node_modules/doppler-gpu/src/formats/sha256.js
node artifacts/bounded-consolidation-2026-09-20/hash-state-read-browser-probe.js /var/tmp/doppler-consolidation-stream-v2-20260920/consumer/node_modules/doppler-gpu/src/formats/sha256.js
```

## Package and boundary effects

`loading-package-audit.json` records the exact changed package entries and the
Node/npm toolchain used for the package-budget check. An unused tooling declaration
forwarder leaves the generated package closure after its consumers switch to the
canonical config declaration. Its repository compatibility file is retained.
The generated demo inventory changes; UI behavior is preserved.

Component: `doppler.runtime-source.client`, `doppler.runtime-source.config`,
`doppler.runtime-source.tooling`. Intent: preserved. Boundary effects: additive
artifact-reader contract and loading policy; no computation, shader, trust,
pooling, or execution-selection changes. Acceptance commands and outcomes are
recorded alongside this file.
