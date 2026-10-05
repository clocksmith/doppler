# Buffer retirement qualification

The pool retains each retired buffer until its own queue completion succeeds or
confirmed device loss makes further execution impossible. A rejected completion
without confirmed loss retains ownership; repeated `destroy()` retries cleanup.
A throwing buffer destructor remains owned and does not start a retry loop.

The original installed generation log reproduces the deferred-destruction warning.
The corrected installed package executes generation, cancellation followed by
reuse, destruction, and a fresh process reopening without that warning. These
controls use the unchanged model, workload and output/stopping oracle. The exact
corrected Doe native library is a declared local overlay; this is host qualification,
not a registry publication or an adoption claim. Pre-execution cancellation does
not establish interruption of submitted GPU work.

[Physical retirement](physical-retirement.json) records exactly-once releases
across separate completion boundaries and device loss. The focused cleanup tests
cover completion rejection, retry, terminal loss, delayed retirement and failed
destructors. [Safety](safety.json), installed audits, and the compressed original
and corrected logs retain the execution identities.

[Package audit](package-audit.json) separates pre-existing unpacked-budget drift
from the changes to the existing runtime, declaration and changelog entries.
No package entries were added. The package-content guard records the exact audited
candidate; the source architecture ceiling is unchanged.

[Manifest](manifest.json) binds the retained files. `green-check.log.gz` contains
`npm run check:green`; after the final whitespace-only source cleanup, the focused
buffer/RoPE tests and `npm run public:boundaries:check` pass again. Reproduce with
those commands from the repository root. The Doe campaign report retains native
custody and the complete installation replay inputs.
