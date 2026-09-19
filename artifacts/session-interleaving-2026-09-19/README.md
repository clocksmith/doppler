# Session interleaving regression evidence

Baseline: `d27bfbaf` on `main`. No branch reconciliation or historical signed
artifact is changed by this work.

The new concurrent-close tests failed on the baseline: a second `close()`
reported completion before unload finished and suppressed its failure. One
retained cleanup task now makes close and asynchronous disposal await the same
outcome, including synchronous host exceptions. Cleanup is never retried
implicitly, and the session rejects new work as soon as closing starts.

Focused regression command:

```sh
node tools/run-node-tests.js tests/inference/pipeline-session-interleaving.test.js tests/inference/pipeline-context-restoration.test.js tests/inference/pipeline-session-config.test.js tests/integration/doppler-scoped-session.test.js tests/storage/model-read-session.test.js tests/gpu/device-ownership-contract.test.js tests/gpu/shader-source-scope.test.js
```

All seven files pass. The new factory tests use explicit promise gates to
exercise cancelled loading with shared and separate devices, cleanup throwing
without hiding cancellation, shader/config restoration, pooled buffer reuse,
stream cleanup throwing with another request queued, and loss of device A while
B continues. Existing storage tests cover cancellation and closing A while B
loads and reads the same shard filename from its own store.

The complete unit suite contains 844 files. Repository verification command:

```sh
npm run check:green
```

The fresh complete check passed after regenerating the dependency view and
updating the audited package limits; all 844 unit files passed.

These are contract tests with resource doubles, not physical GPU qualification.
No numerical implementation, model support claim or performance claim changes.

The [package audit](package-audit.json) records the retained candidate archive.
Comparing extracted baseline and candidate archives finds exactly two changed
files: the scoped-session implementation and its declaration. File inventory
remains 1,858 entries. Node 22.23.2/npm 10.9.8 measures 2,149,978 compressed bytes
and 11,302,207 unpacked bytes. Packed-module checks also pass for pending close,
concurrent disposal, exactly-once unload and shared failure outcomes.

Component: `doppler.runtime-source.client`, `doppler.tests`,
`doppler.repository-tooling`, `doppler.docs`.

Intent: preserved.

Acceptance evidence: the focused command above, repository check chain and
package audit.

Boundary effects: none; existing dependency and package boundaries remain intact.
