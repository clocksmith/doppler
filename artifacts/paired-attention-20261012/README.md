# Paired attention controls

Component: `doppler.runtime-source.gpu` and `doppler.runtime-source.inference`.
Intent: preserved; explicit approved two-F32 intermediate representation.
Boundary effects: paired attention producer, direct F16 KV storage, attention readout.

`mac.json` and `amd.json` each pass 29 actual public-wrapper controls. These cover exact QG/QKV bit permutations, source-epsilon Q/K normalization, paired RoPE tables and partial rotations, and causal GQA with F16 cache operands. They include immediate/recorded dispatch and owned/borrowed outputs. The largest attention error is below 1.3e-15 against independent F64 equations; long-position table error is below 2.3e-11 against the frozen 2e-10 bound. Each receipt binds its served kernel/registry inputs and actual browser/adapter. Both report no active or deferred pool owners before pool/device shutdown; reusable pool bytes are separately reported.

Run `node tests/kernels/paired-attention-physical.js <receipt.json>` on a physical F16-capable Chromium adapter. These operator controls do not establish the full layer, model, installed archive, or numerical release gate. The separate `paired-attention-layer-physical.js` fixture exercises the actual full layer interpreter and is pending its first physical result at this checkpoint.

Retained failures:

- `mac-initial.json`: fixture supplied a non-F32 explicit inverse frequency, correctly rejected by the existing frequency contract. The fixture now declares its F32 value, and the independent reference uses that value.
- `mac-routing-failure.json` and `contracts-routing-failure.log`: actual ordinary attention wrapper missed paired dispatch because the branch was inserted into the different BDPA executor. The branch moved to the ordinary executor; CPU guards and the physical public wrapper were rerun successfully.
- `split-allocation-fixture-failure.log`: injected allocation failure counted device initialization instead of the split outputs. The injection now selects the owned split allocation labels.
- `amd-setup-failure.log`: initial remote checkout omitted an already committed wrapper dependency. It failed before GPU execution. The successful AMD receipt used an isolated source snapshot plus explicit current-file overlay, not an installed package claim.

`split-before.log` retains the original metadata rejection failure. `split-after.log`, `contracts.log`, `cache-cleanup.log`, `legacy-contracts.log`, and `plans.log` retain focused success/failure ownership and scalar compatibility checks. No reference, tolerance, accepted application pin, or production deployment was changed by these controls.
