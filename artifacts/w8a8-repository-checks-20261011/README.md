# W8A8 repository checks

The subsequent complete `npm run check:green` invocation **passed** at source
`e79513e3`: **917 unit files**, including the **36 Capsule files** reused by the
same invocation. Every recorded implementation and configuration hash remained
unchanged through completion; concurrent commits contained evidence only.
`completion-check-green-index.json` binds the full log, source hash inventory,
and ending audit. This is repository acceptance, not model or release acceptance.

The earlier retained runs below remain historical evidence.

The corrected unit sweep passed **917 test files**. The separate Capsule sweep
passed **36 files**. All other `check:green` script gates passed except a stale
generated dependency map; regenerating that map and its check then passed.
Final decode-output ownership, seeded decode ownership, and pipeline-unload
controls passed against the corrected runtime.

The original `check:green` invocation failed four unit files and three other
script gates. Its complete output remains here. Two unload test failures were
new regressions introduced during that sweep before their repair. Support and
routing inventories were regenerated, W8 dispatch geometry was declared, and
owned cleanup was repaired. See `summary.json` for exact paths and commits.

This is **combined check coverage**, not a clean-checkout, immutable-source
`npm run check:green` success. Shared source changed during the sweeps; the final
affected cleanup controls and generated checks are retained separately. A
launcher error that ran no checks is also retained. Logs are losslessly gzip
compressed; the summary records compressed and original byte digests.

Component: `doppler` and `doppler.docs`. Intent: preserved. Boundary effects:
none. This evidence establishes repository checks only. It does not establish
model correctness, task quality, physical GPU performance, or release acceptance.
