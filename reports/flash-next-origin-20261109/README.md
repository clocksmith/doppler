# Flash-Next upstream drift and GDN softplus repair

This investigation preserves the frozen source, artifact, inputs and
`doppler.boundary-tolerance/quantized-v1` acceptance policy. It separates
constructive arithmetic witnesses, substituted-execution diagnostics, physical
operator tests and ordinary full-network candidate results. None alone qualifies
the model. The original failed runs remain failed.

## Trace backward from Layer 25

`origin-protocol.json` binds the original 2,304-token geometry, three inspected
positions, weights, source and GPU captures. `origin-witness.json` chains the
first hyper-connection from the exact embedding input without resetting to
captured intermediates. All 101,784 sampled GPU values across eight stages match
the modeled arithmetic bit-for-bit. One-ULP corruption, a changed reduction tree
and a missing norm offset are distinguishable negative controls.

The first observed difference at positions 1717 and 2271 is the first down
projection, on identical normalized inputs. That does not establish an incorrect
projection: the modeled GPU reduction is closer to the exact dot product at
inspected elements than the source result. The source's F32 accumulation order
differs. A witness proves the arithmetic it models, not permission for cumulative
model error.

Fresh physical full-network counterfactuals test whether that first difference
causes the final failure:

| Execution | Maximum logit difference | Violations under unchanged policy |
| --- | ---: | ---: |
| Frozen F32-KV baseline, no substitution | 0.10902119 | 13 |
| Source first down-projection output substituted | 0.10902023 | 13 |
| Source complete first HC branch input and gate substituted | 0.10903740 | 13 |
| Source complete state entering Layer 24 substituted | 0.00011677 | 0 |

The baseline logits and inspected router boundaries reproduce retained bytes.
The first two substitutions retain both Layer-25 expert-set differences, at
1717 and 2271. The Layer-24 substitution eliminates them, although another expert
set differs at Layer 43, position 735. These are diagnostics, **not passing
normal-model executions**. See `controls-comparison.json` and
`compare-controls.py`.

## Locate a local defect rather than inherited error

The source is resumed from its frozen checkpoint with original batch geometry.
Replaying source branches first verifies the controls independently:

- Layer 23: sampled source branch outputs are bit-exact. Full-sequence attention
  on exact GPU branch inputs has relative RMS difference 1.91e-6; inherited input
  differences dominate the total. QSA selections at the three inspected positions
  are unchanged. See `layer23-same-input.json`.
- Layers 20–22: each complete source GDN branch reproduces all 5,898,240 values
  bit-exactly. On exact GPU projections, core relative RMS differences are
  7.55e-7, 9.15e-7 and 4.21e-6 respectively. Layer 22 is the local outlier;
  its output projection remains at 3.92e-7 on exact core input. See
  `linear20-22-controls.json`.

Layer 22's highest-error value head has very small negative decay arguments.
The scalar kernel computes softplus as `log(1 + exp(x))`, with the addition
rounded to F32. It loses the small exponential before the logarithm. At **x =
−20**, the original kernel returns **0**, whereas the independently rounded
binary64 reference is **2.06115369216775e-9**. The shortcut used for x < −20
therefore also creates a discontinuity at the boundary. This is cancellation in
the implementation, beyond merely choosing a different summation order.

## Repair and independent operator checks

For small exponentials, evaluate the log1p Taylor series through its fifth term.
At e ≤ 1/64, its omitted sixth term is below a binary32 ULP. For the remaining
negative inputs, use `2*atanh(e/(2+e))`, whose argument never exceeds 1/3, with
the odd series through degree 19. The omitted tail is below 1.1e-11. Positive
inputs and the existing source thresholds retain their previous behavior.

The physical test extracts the actual scalar functions from the shaders, then
compares with binary64 `log1p(exp(x))`, rounded once to F32. Its six-ULP bound,
positivity and monotonicity requirements were fixed before the repair. It covers
8,226 inputs, including threshold neighbors, and does not compare with another
copy of the proposed polynomial.

The original eight inspected surfaces each fail at 1,151 inputs; at −20 they
return zero. The repaired 17 surfaces pass on physical Apple Metal and AMD
Vulkan. The first stable atanh candidate stays within three ULP; the final
small-input Taylor refinement stays within two ULP. Shader digests distinguish
the candidates. These are scalar operator results, not browser/model support.

On fixed captured Layer-22 projections, physical ordinary-F32 execution with
the first stable formulation reduces full core relative RMS error from
4.21e-6 to 1.72e-6 and maximum error from 4.38e-5 to 1.24e-5. No source tensor
is injected into this operator's computation.

An exploratory two-component-state control was unsuccessful: the physical
device was lost, so it has no valid candidate numerical result. Kernel logs
recorded an automatic AMD ring recovery. No manual reset, root setting or
service restart was performed. Subsequent ordinary-F32 operator and full-network
runs completed. The existing provider process remained running; that is not a
fresh test of its resident inference session. This repair uses ordinary F32
state, not the unsuccessful precision path.

## Ordinary full-network candidate

The candidate copies the frozen runtime into an isolated operator-local
directory, changes only scalar softplus bodies and declared shader digests, and
retains the same weights, F32-KV reference, execution choices and tolerance.
`prepare-softplus-candidate.py` records the changed source identities. The
original artifact manifest is never rewritten. Registered execution digests in
source recipes are refreshed for future preparation; existing published
artifacts and services are not upgraded.

The first stable atanh candidate reduces the 13 violations to **1**, with
maximum absolute error **0.05957747**, versus 0.10902119 before. It still fails.
Its remaining violation is output token 96693; its error is 0.05908415 against
an unchanged limit of 0.05205551. Layer 25's ranked router logits still differ
at position 1717; position 2271 is repaired. See
`softplus-atanh-comparison.json`. The comparator labels inferred top-k from
captured logits separately from direct GPU expert-index observation.

The final small-input refinement also has **1 violation**, with maximum absolute
error **0.05958652**. See `softplus-comparison.json`. At position 1717, experts
493 and 330 now have equal captured F32 logits. Their exact dot-product margin
on the GPU's actual input is −6.73e-8, while the margin on the source input is
+1.51e-6. At 2271, exact margins on both inputs favor the same source expert.
Thus improving router accumulation alone still cannot reconcile the remaining
choice. See `softplus-router-margin.json` for exact dyadic fractions.

The repaired input trace covers all 26 MLP-router boundaries through Layer 25,
with source RMS and relative error for every layer and inspected position.
Full-sequence relative RMS remains between 1.38e-6 and 3.37e-6; at position
1717 it is 2.65e-6 at Layer 24 and 2.64e-6 at Layer 25. The large Layer-22 local
outlier has been reduced, but these mixed branch inputs do not identify every
residual-stream operation. See `repaired-input-trace.json`.

A separate 15-token short-input check passes all 248,320 prefill logits under
the unchanged policy, with maximum difference **2.38e-5**. Three generation rows
are finite and cleanup has no errors. Only prefill is independently compared;
this is not independent decode-logit parity or answer-quality evidence. See
`softplus-short-comparison.json`.

**Unresolved:** the one long-input violation remains a failure. Further work
must isolate the remaining input difference or justify it independently; this
report neither proves it unavoidable nor authorizes another acceptance policy.
No recipe is promoted on a component improvement, matching first token or
counterfactual pass.

## Reproduction

The binary captures and learned weights remain in the operator's provisioned
data volume. Compact reports bind their hashes, shapes and producers. These
files are not a self-contained model download.

```sh
node --test tools/analyze-hc-origin.test.js tools/layer25-wgsl-witness-core.test.js
node tests/kernels/gated-delta-softplus-physical.test.js OPERATOR_RECEIPT.json
python3 artifacts/qwen38-flash-next-20261013/prepare-softplus-candidate.py \
  FROZEN_RUNTIME BASE_RUN.json REPLACEMENT_SHADER.wgsl \
  tests/kernels/gated-delta-softplus-physical.test.js NEW_RUNTIME NEW_DATA
```

Run the provisioned learned-model harness from `NEW_RUNTIME`, with
`DOPPLER_FLASH_NEXT_PROBE` naming `NEW_DATA/run.json`. Serialize physical runs
with the operator's GPU lock. `compare-softplus.py` checks the bound reference,
policy and candidate captures; `assess-router-inputs.py` computes exact dyadic
margins on each side's actual captured input. Neither alters acceptance.

The same-input controls are reproduced by
`check-f32-attention-same-input.py`, `check-f32-linear-stage-inputs.py` and
`tools/run-gdn-same-input.js`, with explicitly provisioned paths. The first-HC
fixture is rebuilt by `prepare-hc-origin-control.py`; the full-network
substitutions use `tools/run-hc-origin-counterfactual.js` and explicit diagnostic
protocols. They are not production runtime entry points.

The source demo shell independently exceeds its unchanged byte budget:
10,006,898 > 10,000,000 bytes. Comparing every listed shell file with the base
commit finds zero byte growth from this task. No budget was raised and no demo
or runtime was deployed.
