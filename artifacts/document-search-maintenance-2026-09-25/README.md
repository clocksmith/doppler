# Search starter maintenance and Node acceptance

The accepted browser release from September 23 is unchanged. Its vendored runtime
still hashes to `625b8ed8c8e86c28a5b49e1e5ce43969885acd3df309f456eba719d3eeaa5539`.
Browser distribution acceptance and repository hygiene are separate results.

## Runtime hygiene

The five failing gates shared an accidental public export of the internal layer
partition contract. Root/runtime imports no longer acquire that dependency.
The internal implementation now has matching declarations and is strictly
typechecked; missing optional shape constraints, owned serialization, and
non-finite numerical comparisons have regression coverage. No checker budgets,
ownership rules, or unchecked-type allowances were increased.

The installed consumer tests root, runtime, host and generation imports,
declaration visibility, forbidden internal imports, runtime closure, and package
exclusions. The [prior archive fails](prior-runtime-regression.json); the
[corrected archive passes](corrected-runtime-installed-check.json). Its SHA-256 is
`73c0d1227f5d64e2e60d78b2a14d4ecb5b90535d932cf7b184d24eaa466c6b3e`.
The [full repair check](hygiene.json) passed all gates and 863 test files before
the Node extension. The accepted browser archive was not replaced. The [final repository check](repository-final-green.json)
passes every gate and all 864 test files after the Node extension.

## Node

[Download the qualified Node starter](https://huggingface.co/clocksmith/rdrr/resolve/fa3325d41d43bf7ec3a9dcd39e77587fefabdd89/document-search/node/releases/0.1.0/caa1f395d8a0943d9cd62442f87860a1ccbe6e5abb2eb3b5ff54375e4b1e1a6e/doppler-node-document-search-0.1.0.tgz).
Archive SHA-256: `caa1f395d8a0943d9cd62442f87860a1ccbe6e5abb2eb3b5ff54375e4b1e1a6e` (8554262 bytes).
The runtime candidate is also published separately in the same immutable release
folder. This does not update the browser archive or publish a new npm version.

The separate Node application reuses `search.js`, `controller.js`,
`document-store.js`, and `installation.js`. The added runner owns provider lifetime
and filesystem storage. It exposes repeated interactive queries, cancellation,
atomic document/index saves, explicit close, and restart. A failed commit initially
reported its original I/O failure again as a cleanup error; the filesystem adapter
now preserves that original error and has a regression through the shared writer.

Forge reuses the exact retained Program Bundles, shader bytes, weights, tokenizer,
and ModelIR evidence. Passing installed Node source-reference probes supply the
additional execution surface; new signed releases have their own identities.
The browser Capsules and their release history are unchanged.

The final archive is extracted outside the checkout and installed with frozen
`npm ci`. Qualification verifies the application inventory, vendored archive,
installed runtime assets, provider lock and signed models. No checkout runtime
or application source is copied into that consumer afterward. Cold installation
uses all 32 public immutable shard URLs. The retained corpus and source references
are unchanged. The [build receipt](node/build-receipt.json) and
[provider binary identity](node/provider-binary.json) bind the executable inputs.

The [final installed acceptance](node/release.json) passes clean public acquisition,
cold indexing, repeated searches with both models resident, unchanged-document
reuse without GPU submissions, submitted-work cancellation, supersession, atomic
save failure, corrupted-model repair, explicit closure, and device-loss recovery.
A fresh process with kernel-denied IPv4/IPv6 sockets passes offline reopening and
result parity. The interactive CLI handles two queries and exits after EOF; its
installed API refuses to take ownership of an existing WebGPU provider.

Observed final qualification: installation 223,935 ms, cold indexing 2,220 ms,
first query 974 ms, subsequent-query median 1,004 ms, and offline reopening
50,907 ms. Process RSS high-water is 3,266,555,904 bytes; peak requested GPU buffers
are 3,989,755,736 bytes. The repository check overlapped part of qualification;
these are observed diagnostics, not a controlled performance comparison.

The [Node guide](../../examples/document-search/NODE.md) is the consumer path;
the [engineering guide](../../examples/document-search/ENGINEERING.md#build-and-qualify-the-node-application)
contains reproduction steps. GPU qualification is Linux x64, Node 22.22.1,
`webgpu` 0.4.0, AMD Radeon 8060S/RADV Mesa 26.0.3 on the 122 GiB host. Storage in
these physical runs is tmpfs. Process RSS and requested GPU allocations are
reported separately; neither is a minimum-memory or physical-residency claim.

## Remaining evidence

The [browser opening analysis](browser-reopening-baseline-analysis.json) separates
the retained validation and preparation phases without claiming a new run or an
optimization. Disk-backed browser acceptance, a smaller-memory GPU machine,
reboot persistence, and measured opening optimization remain outstanding. The
persistent volume has insufficient free space for the model bytes plus profile.
Injected storage failures do not qualify an actually exhausted volume or reboot.

A [targeted invitation](independent-integration.json) addresses Paper Atlas's real
browser-search feature and explicitly acknowledges the model-download mismatch.
No reply, independent integration, or second revision has been received. The
invitation is not adoption; payment is not a requirement.

Component: `doppler`, `doppler.runtime-source.inference.pipelines.text`, `doppler.repository-tooling`,
`doppler.tests`, `doppler.docs`. Intent: preserved. Boundary effects: accidental
root/runtime partition exports removed; application-specific Node loading and
storage remain outside Runtime. Acceptance evidence: linked installed consumers,
source-reference reports, physical application receipts, and repository checks.
