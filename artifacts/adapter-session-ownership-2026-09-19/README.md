# Adapter/session ownership regression evidence

Baseline: `f86d97cd` on `main`. No branch, numerical implementation, historical
signed artifact, or model support claim changes.

The original seven regression cases failed on the baseline: adapter activation
and resets could change an active stream, and unloading could finish before
adapter preparation restored loader metadata. Two additional compatibility
provider cases reproduced registry mutation on rejected activation and a close
error masking the original execution failure during adapter restoration.

The existing pipeline scope now owns execution and adapter exclusion across
handles for the same pipeline. Preparation cannot activate after close or device
loss. Closing seals immediately, drains active preparation/execution, then
unloads once. Composite evidence operations retain ownership through receipt
construction; internal reranking resets remain permitted. The compatibility
provider commits its registry only after activation and restores temporary
adapters before the operation releases ownership.

Focused regression command:

```sh
node tools/run-node-tests.js tests/integration/lora-session-ownership.test.js tests/integration/dream-provider-contract.test.js tests/integration/doppler-generation-evidence.test.js tests/integration/doppler-rerank-evidence.test.js tests/integration/doppler-scoped-session.test.js tests/inference/pipeline-session-interleaving.test.js
```

All six files pass. The new suite has 16 ownership cases (plus the source
runner's initialization sentinel). It covers active and queued execution,
alias handles, close during preparation, loader failure, device loss, unchanged
adapter/cache/evidence on rejection, call-time generation settings, and provider
registry/restoration. Promise gates establish ordering without timing guesses.

Final repository verification:

```sh
npm run check:green
```

The fresh full chain exited successfully on the final implementation, including
all 845 unit files, architecture/dependency/style checks, strict boundary types,
and installed-consumer/package checks. The local npm 9 archive measured
2,147,185 packed bytes with the same 1,858 entries and 11,307,872 unpacked bytes;
the retained audit below uses the exact Node/npm versions of CI.

The same 16 cases pass against the extracted npm package by replacing the test's
`../../src/` imports with the candidate's absolute `file:` source root and loading
the resulting test as a data URL with the repository Node test bootstrap.
These are resource and computation doubles, not physical GPU qualification.
Consumers must finish or return paused streams before awaiting close; the
change does not force-interrupt adapter I/O or submitted GPU commands.

The [package audit](package-audit.json) records the final candidate archive and
SHA-256. Comparing the extracted baseline and candidate finds eight changed
payload files, no inventory changes, and 5,665 additional unpacked bytes.
Node 22.23.2/npm 10.9.8 measures 1,858 files, 2,151,228 compressed bytes and
11,307,872 unpacked bytes. Package limits reflect that measured payload, without
an added allowance. The archive remains local; this is not an npm publication.

Component: `doppler.runtime-source.client`,
`doppler.runtime-source.client.model-host`,
`doppler.runtime-source.inference.pipelines`, `doppler.tests`,
`doppler.repository-tooling`, `doppler.docs`.

Intent: preserved.

Acceptance evidence: `npm run check:green`, focused source tests, packed
ownership tests, and the package audit above.

Boundary effects: none; existing ownership/dependency policies remain intact.
