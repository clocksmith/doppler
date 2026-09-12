# Demo streaming review

Status: implemented in the working trees, uncommitted and not deployed.
This builds on the earlier [UI review](../demo-ui-2026-09-12/README.md).
`source-digests.json` identifies the current bytes in Doppler and D4DA; the older
review's digests do not certify this streaming update.

- Component: `doppler.demo`, `doppler.runtime-source.client.model-host`,
  `doppler.runtime-source.inference.pipelines.text`; supporting tests, docs and tooling.
- Intent: preserved.
- Acceptance evidence: commands and retained logs below.
- Boundary effects: additive inspection token events; first-token callback repair;
  D4DA's generated demo/wrapper and the same three-file runtime patch. No model,
  kernel, sampling, batching, benchmark, catalog, Doe, publish or deployment changes.

## Behavior

Inspection emits ordered token IDs while generation is pending. The callback
does not decode text or request GPU readbacks. The demo accumulates IDs, decodes
the sequence at most once per animation frame, and changes only the suffix of
the existing answer text node. Decoding the sequence preserves tokenizer context
and incomplete Unicode; an unfinished replacement character is withheld until
more tokens or the final receipt arrive. This batches browser work but does not
claim a constant-time tokenizer or a measured inference speedup.

Normal completion preserves the current answer element and reconciles with the
receipt. Word quality and X-Ray remain completion-time evidence. Stop cancels the
queued frame, flushes received text, preserves the partial answer in history and
keeps receipt export disabled. Errors also preserve received text. Late events
cannot overwrite settled answers. Scrolling up disables automatic following;
the streaming journey performs no navigation or page reload.

The token-ID generator previously omitted its first token from `onToken`.
The repair covers first-token EOS, ordinary decoding, callback failure cleanup,
and successful retry without changing token selection.

## Validation

- `node tools/run-node-tests.js tests/inference/generate-token-ids-behavioral-parity.test.js tests/integration/doppler-generation-evidence.test.js tests/client/inspection-contract.test.js tests/demo tests/integration/demo-surface-contract.test.js`
  passed all 10 files: `unit.log`.
- `npm run test:demo:contract` passed: `contract.log`.
- The same browser command with `DOPPLER_DEMO_ORIGIN=http://127.0.0.1:8075`
  and `DOPPLER_DEMO_ENTRYPOINT=/doppler/` passed the D4DA wrapper:
  `world-contract.log`. The preview served `sites/d4da-com` and allowed the
  service worker's `/doppler/` scope. The preview is now stopped.
- Browser tests explicitly pause generation to inspect partial text, verify DOM
  identity before/after completion, exercise Stop/error/next-turn history, inject
  late events, preserve a reader's scroll position, and count navigation events.
  A controlled animation-frame queue verifies one scheduled paint per token
  burst, Unicode repair, synchronous abort/final flush, and canceled frames.
- The API integration test verifies events arrive before resolution, identical
  final evidence/fingerprints with and without streaming, unchanged observation
  options, invalid callback rejection, aborted completion rejection and ignored
  late callbacks. These tests inject model execution and do not prove GPU inference.
- `hosted-inspection.js` and `hosted-generator.js` are focused extractions of
  those canonical cases with imports redirected to the actual D4DA runtime.
  Both passed: `hosted-runtime.log`. Node reported its existing inferred-module
  warning for D4DA's browser source; this did not prevent execution.
- `npm run typecheck:source`, `npm run source:architecture:check`,
  `npm run source:style:check`, `npm run inference:boundaries:check`,
  `npm run api:docs:check`, `npm run demo:reachability:check` and
  `npm run demo:shell:check` passed; retained logs use the corresponding names.
- D4DA's `npm run check:world:doppler` and hosted shell generator `--check`
  passed: `world-check.log`. `git diff --check` passed in both repositories.

Browser and runtime tests use mocked execution. Live GPU streaming is unverified;
the prior hardware attempt failed downloading all six model shards with HTTP 429
([retained failure](../demo-ui-2026-09-12/hardware.log)). It was not repeated here.
No physical-device rendering budget or model throughput result is claimed.

The initial browser harness expected a newly rebuilt historical assistant node
and timed out after completion. Its assertion now checks the completed live
answer; subsequent browser checks passed and also assert that the original
answer and history nodes survive completion.

## D4DA synchronization

`runtime.patch` is the exact canonical runtime diff applied to D4DA with
`git apply --directory=sites/d4da-com/doppler`; it was checked before application.
It changes only `src/client/model-host/model-session.js`, its declaration, and
`src/inference/pipelines/text/generator/decode-runtime.js`.
The demo was then copied with `npm run sync:world:doppler -- --demo-only`, which
regenerated the shell against the retained hosted runtime. The broader runtime
differences between the repositories were not imported. Versions remain 0.6.0.

## Pre-push validation after the workspace pull

The subsequent `rdpull` advanced canonical Doppler to `04db564d` (0.6.1).
The reviewed UI and streaming source bytes survived unchanged. Its shell was
regenerated for the pulled source, and the 10 focused test files, browser contract,
shell check, and source type check passed again. D4DA's required type check also
passed using the same exact vendored dependencies described in the UI review;
the temporary dependency links were removed afterward.

The new logs are prefixed `prepush-`; `prepush-source-digests.json` records these
bytes separately from the original capture. D4DA remains the reviewed 0.6.0
runtime plus the three-file streaming patch, and its shell check passes.
`npm run check:world:doppler` now fails its full-sync version comparison because
the sibling source is 0.6.1. This push does not claim a full runtime synchronization
or deployment, and the check was not weakened to hide that difference.

During `rdpush`, upstream advanced again to `31bdf3fd`, including a runtime stats
module change. The UI commit rebased cleanly to `21e2c5f1`; D4DA pushed as
`dd9a0c7`. The canonical shell was regenerated for that additional upstream
change. `final-shell.log` and `final-contract.log` retain the passing shell and
browser checks for that final graph. This follow-up updates generated cache
evidence only; the reviewed streaming implementation is unchanged.
