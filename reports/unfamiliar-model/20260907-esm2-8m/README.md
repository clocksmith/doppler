# ESM-2 8M installed encoder qualification

[Acceptance](acceptance.json) binds clean commit
`94570b438002ea681b7114dfb5bb4594f1956e40`, runtime archive SHA-256
`167bfda427e337d8dd2f90fee45dda49d1d6cd32eae510d3729cae67633451e4`,
the pinned MIT source, all reference elements, fresh acquisition/conversion,
installed Node execution, and actual device-loss recovery. This is raw RDRR
encoder qualification on AMD Radeon 8060S, not signed Capsule promotion,
masked-language-model execution, biological validation, or independent adoption.

The frozen corpus has 26, 22, and 50 tokens. Every one of the 31,360 token
embedding elements and 960 pooled elements is checked before and after recovery,
with the original 0.001 absolute tolerance. The evidence archive retains the
initial failures, numerical GELU probe, shader and activation-binding repairs,
candidate packages, clean-checkout acceptance, remote CI, and six manual
interventions. The CPU reference replay is byte-identical.

Reconstruct from an ordinary checkout with Node 22.22.1 and npm 9.2.0:

```bash
npm ci
mkdir /var/tmp/esm-reproduction
tar -xzf reports/unfamiliar-model/20260907-esm2-8m/reproduction.tar.gz -C /var/tmp/esm-reproduction
TMPDIR=/var/tmp node tools/restore-capsule-baseline.js /var/tmp/esm-reproduction /var/tmp/esm-restored
npm --prefix /var/tmp/esm-restored/consumer install --save-exact webgpu@0.4.0 --ignore-scripts --omit=optional --no-audit --no-fund
```

Use new directories on every attempt. The recipe includes the exact runtime
archive and manifest; it downloads six source files at
`facebook/esm2_t6_8M_UR50D@c731040fcd8d73dceaa04b0a8e6329b345b0f5df`, verifies
their hashes, invokes the installed converter, and verifies regenerated shard
bytes. Model weights are obtained from that public origin, not this repository.
The source card declares MIT. The archive carries the source metadata and recipe;
no signing key or new release authorization is included.

Extract `evidence.tar.gz`, copy `node-restored-final-94570b43-config.json` to a new
file, and change only `packageBundlePath`, `modelDir`, `reference.path`, and
`outputDir` to your reconstructed paths. Then run:

```bash
TMPDIR=/var/tmp node tools/qualify-installed-sequence.js /path/to/replay-config.json
```

That retained qualification requires a physical AMD Vulkan device. Another
device needs its own explicit qualification policy and produces separate evidence.
The provider dependency lock is retained under
`restored-final-94570b43/consumer/package-lock.json` in the evidence archive.

For independent CPU reference capture, the reproduction archive includes
`retained/reference/capture-reference-portable.py`. Its four arguments are the
retained policy, acquisition receipt, newly downloaded source directory, and a
new output file. The tested environment was Python 3.14.4, PyTorch
2.11.0+cu130 executing on CPU, and Transformers 5.6.2. Matching all reference
bytes is checked separately from conversion and GPU execution.
