# Failed-prefill retirement boundary

The installed packed W8 model cancellation control on archive `79b7c68` aborted
after an actual GPU submission. The operation rejected with `AbortError`, and
active allocations remained unchanged at 67,108,864 bytes across four buffers.
However, 133 retired buffers totaling 13,094,144 bytes remained deferred after
operation rejection and queue completion. Unloading later retired them. The
complete failed physical receipt remains under
`artifacts/w8a8-model-qualification-20261011/installed/`.

The prefill owner called asynchronous recorder abort without awaiting it. Its
failure boundary also did not await the pool's existing explicit retirement
operation. The repair captures the owning device pool before asynchronous work,
awaits recorder abort, then awaits a retirement snapshot on failure. It leaves
resident, active, and reusable resources alone; it does not clear a global pool
or drain buffers retired after that snapshot. Successful transferred hidden
states and explicit GPU-logit ownership remain with their callers.

The existing `prefill-output-ownership.test.js` now gates recorder cleanup and
retirement independently. It changes the global device during cleanup to ensure
the original pool is used, and checks that primary errors remain attributable
when recorder and retirement cleanup both fail. A subsequent call succeeds.
The new boundary assertion fails before the repair and passes afterward;
`before.log.gz` and `after.log.gz` retain those results.

Affected checks:

```sh
node tests/inference/prefill-output-ownership.test.js
node tests/inference/buffer-pool-cleanup.test.js
node tests/gpu/command-recorder-cleanup.test.js
```

The pool controls separately exercise retained active resources, later
retirements, failed destruction, and retry. These are deterministic controls;
the physical qualification below is separate.

The post-repair physical controls passed on exact archive
`47d7d18fb51a0852b6b0c35b0590e306b460c743eefe2d36b6615839c8651edc`,
for both portable and packed W8 models. Actual-submission cancellation rejected
with `AbortError`; active buffers stayed at the same 67,108,864-byte baseline,
and deferred bytes/count were both zero when cancellation completed. Subsequent
generation succeeded; unload settled every pool ownership category to zero.
The receipts are
[`w8a8-packed-installed-model-03.json.gz`](../w8a8-model-qualification-20261011/installed/w8a8-packed-installed-model-03.json.gz)
and
[`w8a8-installed-model-06.json.gz`](../w8a8-model-qualification-20261011/installed/w8a8-installed-model-06.json.gz).
These lifecycle successes do not change the model's failed task-quality result.

Component: `doppler.runtime-source.inference.pipelines.text`. Intent: preserved.
Acceptance evidence: the focused commands and retained logs above. Boundary
effects: failure completion awaits the existing memory-retirement contract.
