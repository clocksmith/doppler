# Q/K regression and Linux qualification

Component: Doppler arithmetic and its ordinary package. Intent: qualify the
retained numerical candidate for Reploid without changing precision, reference
inputs, or the full-generation tolerance of 0.001.

Source candidate: `d8a17066`, version 0.6.19. This work adds the missing physical
Q/K runner; it does not change production arithmetic.

`qk-normalization-linux.json` executes the complete canonical `rmsnorm_qk.wgsl`
on physical AMD RDNA 3 through Chrome/Vulkan. Captured Q/K operands and head
sizes 64, 128, 256, 512, and 1024 pass the existing RMSNorm tolerance. Receipts
retain exact shader, fixture, input, and output hashes and browser identity.
Independent captured-operand maximum absolute error is 6.44760784318521e-7.
This is operator evidence, not generation acceptance. The Mac/Metal run now
passes too: all six cases have identical output bits on both physical GPUs.
`qk-platform-comparison.json` retains the checked hashes and host identities.

Run the identical runner on the Mac, using the identical source and fixture:

```
node tests/integration/qk-normalization-platform.js src/gpu/kernels/rmsnorm_qk.wgsl artifacts/recovery-20261009/qk-normalization-mac.json
```

Compare fixture, shader, and input hashes before comparing each case's output
hash and accuracy metrics. Preserve both receipts and any disagreement.

The recurrent scalar and Q4K fused SiLU controls pass. The generic fused FFN and
batched prefill SiLU controls exceed their stricter diagnostic accuracy bounds;
their unfavorable raw receipts remain retained. These controls do not establish
that those generic kernels own Reploid's remaining numerical disagreement.

The initial full CPU/check chain found stale routing/dependency inventories and
an outdated package inventory ceiling. Generated inventories were refreshed.
The package ceiling now records the measured 1,897-entry inventory, including
four captured-operand fixtures required by the installed kernel suite and already
shipped in 0.6.15 and 0.6.18. No numerical acceptance bound changed. The routing
audit retains 119 surfaced integrity failures across unrelated local manifests;
refreshing the audit does not qualify them. `package-inventory.json` retains the
exact measured payload and the four-file delta from the prior inventory ceiling.

The full CPU suite passes all 895 test files. The three initially failing
metadata gates pass after correction; `cpu-and-gates.json` links those categories
to their retained logs. Installed-package full-generation acceptance remains
required before publication/deployment acceptance. The last complete 0.6.18
physical comparison remains failing; its frozen inputs must not be regenerated.

The complete standard 0.6.19 physical comparison also fails: 46/110 steps exceed
0.001, maximum 0.0018558502197265625. Reploid retains its result under
`artifacts/recovery-20261009/numerical-019-physical-result.json`. The first
three layer outputs match; detailed layer-three operation capture is the next
diagnostic. The identical isolated Q/K results do not close this remaining gap.

## Investigation checkpoint: 0.6.21 candidate

The first remaining operation in the layer-three physical trace is RoPE.
Identical captured projection and Q/K normalization values rotate differently
on AMD/Vulkan and Apple/Metal. Refined frequency/trig calculation and explicit
rotation arithmetic produce identical prototype values and smaller captured
Float64 error. See rope-prototype-summary.json and rope-captured-inputs.json.
These operator results do not qualify the installed package. Its physical
110-comparison gate is running, unchanged at 0.001. CPU qualification remains
incomplete; generated closure, package budgets and digest-pinning tests require
updates. No npm publication or application deployment was performed.
