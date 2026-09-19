# Source rotary frequency correction

The frozen full-output references exposed source frequencies that conversion had
discarded. The retained first-layer readbacks and source counterfactual locate the
first substantive numerical divergence at rotary encoding. Conversion now retains
the exact source vectors in the manifest and ModelIR; the GPU precompute program
uses those values. Validation rejects conflicting or unsupported source data, and
cache identity distinguishes source values from generated frequencies.

`evidence.json` hashes the diagnosis, full CPU references, GPU comparisons, and
package inventory. The earlier GELU correction belongs to the base revision;
the new correction preserves the checkpoint's rotary frequencies. Earlier signed
artifacts remain unchanged. This is numerical correctness evidence on the recorded
AMD Vulkan host, without a performance or independent-operator claim.

With dependencies installed in a clean checkout, provide the exact source files
named and hashed by `retained-control/full-reference.json`, an unused output
directory on a volume with space for converted weights, and a Vulkan-capable
Chromium executable:

```sh
TMPDIR=/path/to/ssd/temp node reports/sequence-source-frequencies/20260908/reproduce.js --source /path/to/checkpoint --out /path/to/new-output --browser /path/to/chrome
```

The command checks source and reference bytes, reconverts, and runs each complete
output comparison in a fresh browser process. It cannot substitute another model
or widen the frozen tolerances. The reference is an independent source
implementation; the retained run was performed by the same local operator.

Component: configuration, converter, text pipeline, GPU rotary precompute

Intent: preserved

Acceptance evidence: `evidence.json`; focused source-frequency and cache tests;
physical GPU table test; full source-output browser qualification; repository checks.

Boundary effects: source-owned manifest and ModelIR frequency fields reach the
existing GPU program; model execution remains on WebGPU. The accompanying
generation-evidence repair includes the resolved presence penalty in its hash.
