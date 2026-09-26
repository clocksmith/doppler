# Rig and Run naming

Component: `doppler`
Intent: preserved
Boundary effects: client API and preparation tooling names; compatibility aliases retained.
Acceptance evidence: [validation](validation.json), [installed package](installed-package.json),
[final gates](final-gates.log), and [compatibility tests](compatibility-tests.log).

Doppler Rig prepares Capsules; Doppler Run executes them. The [migration guide](../../docs/rig-run-naming.md)
describes canonical names and preserved APIs, schema IDs, configuration keys, and paths.
New and legacy Rig calls produce identical signed fixture bytes. The installed
package exercises `doppler-gpu/run`, its old aliases, public types, and synthetic
Capsule execution. These checks do not establish new physical model qualification.

The initial full check passed all 864 test files and failed four inventory/package
gates. After the naming inventories and measured payload allowance were updated,
every non-unit gate passed in a final sweep. Four focused compiler/compatibility
files also passed again. No architecture exception or numerical acceptance was relaxed.
The package remains 1864 files; its added alias/name payload increases the unpacked
allowance by exactly 1771 bytes. The packed-size and file-count limits are unchanged.

Published search archives, model artifacts, signed Capsules, and historical
qualification records are unchanged. The locally tested archive is retained at
`/var/tmp/doppler-rig-run-package-20260926/doppler-gpu-0.6.2.tgz`; it was not published.
