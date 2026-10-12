# Archive9 frozen-source CPU unit sweep

All **962 selected unit files** completed successfully across the full sweep and
one focused environment rerun. **46 files explicitly skip optional cases**; their
messages remain in `file-results.json` and the complete logs. The canonical
pending public-logits file remains excluded. This is CPU contract evidence, not
GPU acceptance, source answer quality, signed opening or release qualification.

The source is `bddea7ef32ce54e0dea14793258a6857db65aad7`, exported outside the live
checkout. The ordinary archive SHA-256 is
`8cae56b0748a5c66add985122d8bbf72b7492e2e0828f713f1b0d7425a59a4e5`.
All 1,974 package files exactly matched that export before execution. All 10,395
tracked source files, archive bytes and installed dependency lock remained
unchanged afterward. The private Git HEAD and tracked diff remained at the
frozen source, including after the focused rerun. See `binding.json`,
`archive-files.json`, `tracked-before.json.gz` and `source-after.json`.

The x128 command was:

```sh
taskset -c 4-7 nice -n 10 node tools/run-node-tests.js --suite unit
```

The canonical runner executes one file at a time. CPU libraries were limited to
four threads. Node was v22.22.1. GPU discovery was explicitly unavailable through
the existing provider override, so optional physical tests used their ordinary
skip behavior. Git metadata/index were private and detached, with no inherited
`GIT_*` variables. The snapshot used a real owned `node_modules` directory whose
entries linked to existing dependencies, plus the exact two sibling Reploid
fixture files recorded in `sibling-binding.json`. No live checkout was changed.

`unit.log.gz` preserves 961 successful files and the one failed synthetic-provider
bootstrap test. The global unavailable-provider override superseded that test's
already-installed synthetic GPU. Removing only that environment variable for
that exact file produced a passing rerun in `synthetic-provider-rerun.log.gz`.
No physical provider or model was used by that fixture. The original failure
remains visible; it was not a runtime repair or a complete second sweep.

`preparation-events.json` separately records an incomplete-transfer hash rejection
and unavailable remote Git authentication. Setup used a local Git object bundle
to complete private metadata; no source test started until all identities passed.
`prepare.py` and `run.py` retain the exact preparation/execution scripts.
`summary.json` and `file-results.json` retain each trial and its actual disposition.
Original uncompressed logs and the frozen export remain at the paths in the binding.
