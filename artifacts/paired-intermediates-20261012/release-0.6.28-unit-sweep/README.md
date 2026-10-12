# Final 0.6.28 frozen-source CPU unit sweep

All **963 selected unit files** completed successfully across one full sweep
and one focused environment rerun. **46 files explicitly skip optional cases**.
The canonical pending public-logits file remains excluded. This is CPU contract
evidence, not physical GPU, model-quality, signed execution or release acceptance.

The source is `5c2b31888f3d98d7cb5fc9284b47df4cb0c18a9d`. The ordinary archive
SHA-256 is `102fd0c4900c6744fb7d103a4b3609365b57e8538657bd47b8dfd8e7ac3469e7`.
All 1,976 package files matched the frozen source. All 10,476 tracked files,
archive bytes, dependency lock, private Git HEAD and tracked diff remained
unchanged after the complete sweep and focused rerun.

```sh
taskset -c 0-3 nice -n 10 node tools/run-node-tests.js --suite unit
```

The canonical runner executes one file at a time on x128. Node was v22.22.1;
CPU libraries were limited to four threads. The source-model document control
ran independently on cores 4–7. These are correctness observations with no
controlled performance claim. The snapshot has private detached Git metadata,
no inherited Git variables, a real owned dependency directory linking existing
dependencies, and two exact sibling Reploid fixture files. The live checkout was
not modified. `binding.json` and `sibling-binding.json` identify every path.

The full sweep passed 962 files and rejected the synthetic-provider bootstrap
fixture because the global unavailable-provider environment override took
precedence over its installed synthetic GPU. The fixture bytes exactly match
the previously reviewed archive9 test. Removing only that override for this
one file passed. Both complete logs and the actual environment delta are
retained; there was no runtime repair or second full sweep.

`selected-files.json` records the exact frozen runner's selection.
`file-results.json` records every trial and explicit skip observation.
`summary.json` reconciles all 963 selected files with successful final results.
`raw-identities.json` binds compressed and original complete logs and the
tracked-source inventory. `prepare.py` and `run.py` retain the preparation and
execution scripts. Private metadata imported an exact locally constructed Git
bundle after source/archive transfer and hash verification, without fetching
into the live repository or changing its branch.

Component: `doppler`. Intent: preserved. Boundary effects: none.
Acceptance evidence: the exact command, source/archive identities, raw logs and
post-run byte verification retained here.
