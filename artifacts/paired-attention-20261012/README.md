# Paired attention controls

Component: `doppler.runtime-source.gpu` and `doppler.runtime-source.inference`.
Intent: preserved; explicit approved two-F32 intermediate representation.
Boundary effects: paired attention producer, direct F16 KV storage, attention readout.

`mac.json` and `amd.json` each pass 29 actual public-wrapper controls. These cover exact QG/QKV bit permutations, source-epsilon Q/K normalization, paired RoPE tables and partial rotations, and causal GQA with F16 cache operands. They include immediate/recorded dispatch and owned/borrowed outputs. The largest attention error is below 1.3e-15 against independent F64 equations; long-position table error is below 2.3e-11 against the frozen 2e-10 bound. Each receipt binds its served kernel/registry inputs and actual browser/adapter. Both report no active or deferred pool owners before pool/device shutdown; reusable pool bytes are separately reported.

Run `node tests/kernels/paired-attention-physical.js <receipt.json>` on a physical F16-capable Chromium adapter. These operator controls do not establish the full layer, model, installed archive, or numerical release gate. The separate `paired-attention-layer-physical.js` fixture exercises the actual full layer interpreter and passes its four Mac cases in `layer-mac.json`: prefill3 then decode1 on the same real cache, each through immediate and recorded APIs. Final output max error is 5.97e-16 at the frozen2e-10 bound, and all stored K/V half words exactly match independent nearest-even F64 producer references. No active/deferred pool ownership remains. The receipt hashes all309 actually served source files. The matching AMD full-layer run remains pending.

Retained failures:

- `mac-initial.json`: fixture supplied a non-F32 explicit inverse frequency, correctly rejected by the existing frequency contract. The fixture now declares its F32 value, and the independent reference uses that value.
- `mac-routing-failure.json` and `contracts-routing-failure.log`: actual ordinary attention wrapper missed paired dispatch because the branch was inserted into the different BDPA executor. The branch moved to the ordinary executor; CPU guards and the physical public wrapper were rerun successfully.
- `split-allocation-fixture-failure.log`: injected allocation failure counted device initialization instead of the split outputs. The injection now selects the owned split allocation labels.
- `amd-setup-failure.log`: initial remote checkout omitted an already committed wrapper dependency. It failed before GPU execution. The successful AMD receipt used an isolated source snapshot plus explicit current-file overlay, not an installed package claim.

`split-before.log` retains the original metadata rejection failure. `split-after.log`, `contracts.log`, `cache-cleanup.log`, `legacy-contracts.log`, and `plans.log` retain focused success/failure ownership and scalar compatibility checks. No reference, tolerance, accepted application pin, or production deployment was changed by these controls.

Full-layer fixture and boundary dispositions:

- `layer-mac-initial.json`, `layer-mac-binding.json`, and `layer-mac-path.json` preserve incomplete fixture state/role/weight-adapter setup. The final fixture uses actual `getNormWeightBuffer`, an explicit declared input-normalization role, and the ordinary runtime statistics/observation fields.
- `layer-mac-public-norm.json` exposed a real integration failure: the interpreter supplies `null` for disabled window/softcap while the paired wrapper rejected those values. `null-disabled-before.log` reproduces it without a GPU. The wrapper now accepts exactly null as disabled; it does not enable any unsupported positive softcap. `null-disabled-after.log` retains the first focused repair.
- `layer-mac-null-fixed.json` passes the first normalization repair. Explicit missing window/softcap policies then received their own failing-before guard (`missing-policy-before.log`) and rejection tests (`missing-policy-after.log`). Final `layer-mac.json` tests those stricter bytes. Legacy scalar defaults remain unchanged. Operator fixtures now declare disabled0 explicitly.
