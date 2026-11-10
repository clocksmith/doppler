# Flash-Next expert summation order: isolated experiment failed

Changing only the routed-expert sum to ascending expert ID does **not** repair
the retained inventory prompt. The unchanged full-logit gate still fails.

| Case | Route-slot baseline violations | Expert-ID candidate violations | Candidate max absolute error |
| --- | ---: | ---: | ---: |
| marbles, 22 tokens | 0 | 0 | 0.0000352859 |
| inventory, 2,187 tokens | 30,664 | **30,616** | **0.4666347504** |

Both greedy tokens match the independent source. The 48 fewer violations do
not establish a useful correctness improvement; the large discrepancy remains.
No production kernel, API, dtype, reference, artifact or tolerance was changed.
This candidate is not promoted.

## What changed

The pinned eager source visits active experts in ascending expert-ID order.
The frozen GPU scatter visits route slots, which follow routing rank. Both
round each weighted output to F32 before adding. The private candidate sorts
the summation order by expert ID while preserving the original expert/output/
weight pairing and the two uniform workgroup barriers per slot.

[preparation.json](preparation.json) identifies five implementation changes
plus the generated shader digest. All 3,094 other source files were verified
byte-identical to the frozen parent. This is a quarantined experiment, not a
production API design: its scatter wrapper requires an indices buffer, and
only the tested non-grouped routed path is wired. No other model or execution
path is claimed to work with this candidate.

[operator-baseline.json](operator-baseline.json) and
[operator-candidate.json](operator-candidate.json) each retain six physical AMD
cases: immediate and recorded execution, a token/workgroup boundary, ten
routes, and the preceding product-rounding cancellation case. All 1,060
outputs per runtime match their respective independent F32-order reference.
Input hashes match between runtimes. The deliberate order case produces 1 in
route order and 0 in expert-ID order, showing sensitivity without claiming
that either order alone establishes model correctness.

[protocol.json](protocol.json) was written before candidate model execution.
[comparison.json](comparison.json) retains the failed full vectors under the
original policy. [model-run-binding.json](model-run-binding.json) binds the
actual changed runtime bytes, raw observation, inputs, reference, comparison
and retained score payloads. All GPU and cleanup error lists are empty.

## Reproduce

Use the retained frozen parent runtime and a fresh candidate directory:

```sh
python3 prepare-candidate.py FROZEN_RUNTIME NEW_DIRECTORY/runtime
node operator-control.js FROZEN_RUNTIME route-slot NEW_BASELINE.json
node operator-control.js NEW_DIRECTORY/runtime expert-id NEW_CANDIDATE.json
```

The preparation was reconstructed independently on CPU with identical output
hashes. Existing-candidate rejection preserved its receipt. The model run used
the preceding new-prompt configuration unchanged, substituting only the
private runtime copy and fresh output paths. Keep model runs serialized under
the existing probe lock and the shared GPU lease. The existing 27B provider
and GPU memory limit were unchanged. These co-resident diagnostic timings are
not a throughput comparison.

The next arithmetic control compares the pinned chunked and recurrent
gated-delta algorithms on common inputs. This failed experiment establishes
neither an unavoidable numerical tradeoff nor an upstream defect.

Component: `doppler.repository-tooling`, `doppler.tests`. Intent: preserved.
Boundary effects: isolated runtime experiment only; no production or acceptance change.
