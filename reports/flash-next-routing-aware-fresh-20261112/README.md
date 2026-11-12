# Flash-Next: routing-aware gate, fresh unforced test

**Status: INCOMPLETE — GPU steps on hold. Flash-Next stays BLOCKED.**
The strict gate, `doppler.boundary-tolerance/quantized-v1`, is unchanged.
The routing-aware policy is still **proposed**. It is not owner-approved, and
nothing is released. The limits were frozen for this test only.

## The two missing pieces (commit `210719b7`)

1. **Same-input operator checks over every layer, the final norm and the LM head.**
   - [`flash-next-same-input-capture-physical.test.js`](../../tests/integration/flash-next-same-input-capture-physical.test.js)
     captures one complete unforced prefill. It records:
     - the hyper-connection boundaries at the policy's sampled positions;
     - each sequence mixer's complete input;
     - the executed top-k;
     - the routed-expert output before the shared expert is added
       ([loader](../../tests/helpers/moe-shared-split-observation-loader.js));
     - the root mixer;
     - the last-position logits.
   - [`audit-same-input-all-layers.py`](audit-same-input-all-layers.py) feeds
     each captured GPU input to the pinned CPU source and scores the result
     against quantized-v1. It covers these families: token embedding,
     hyper-connection, gated delta, full attention, routed MoE, shared expert,
     final norm and LM head.
   - Flash-Next has no final RMSNorm. The root hyper-connection mixer runs in
     its place, so the final-norm family checks that mixer.
2. **GPU task-quality capture.**
   - [`flash-next-task-quality-physical.test.js`](../../tests/integration/flash-next-task-quality-physical.test.js)
     records two things for each case:
     - teacher-forced logits at the sampled positions, taken from the unchanged
       prefill logits path
       ([loader](../../tests/helpers/prefill-logit-rows-loader.js)); the last
       row must be byte-identical to the production prefill logits;
     - the greedy continuation.
   - [`cpu-reference.py`](cpu-reference.py) runs the pinned producer
     (`f6bc0f15…`). It adds two observations only: router rows and
     sampled-position logits. It then runs cache-free greedy rounds.
   - [`build-evidence.py`](build-evidence.py) writes the evidence and
     task-quality files. The evaluator alone decides.

Supporting tools:

- [`run-routing-controls.js`](run-routing-controls.js) runs unforced
  observation and the forced-indices control in one model load. Every new
  Flash-Next GPU process spends about 12 minutes verifying n-gram table pieces
  on this host.
- [`prepare-runs.js`](prepare-runs.js) resolves the named runtime profile.
- [`freeze-inputs.py`](freeze-inputs.py) freezes the prompt tokens.
- [`run-fresh.sh`](run-fresh.sh) runs the procedure.

**Smoke tests.** These ran before the freeze, on the retained non-pinned
`marbles` and `inventory` prompts only.

- The CPU wrapper output is bit-identical to the retained reference: argmax,
  top-20, router records, layer samples and logits.
- The same-input audit found 0 violations across all eight families and 48
  layers (worst ratio to the limit: 1.6e-4).
- The observe, plain and task-quality prefill logits are byte-identical.
- The forced-indices control had 0 violations (maximum error 2.6e-5).
- The CPU and GPU greedy tokens agree: [8646, 220].

## Freeze record (commit `b826337d`)

[`freeze.json`](freeze.json), frozen at **2026-11-12T16:07:59Z**:

- Canonical policy hash:
  `sha256:7a63c40cfbf0753ebe9946d5c74cd458f6fbcd6d726562a47f193e0b2c9bbbd0`
- Policy file SHA-256: `de397c0b…`
- Prompt set: `sha256:d87a15ac…`
- Tooling commit: `210719b7`

All proposed values are used as-is. No rule was found internally inconsistent.

The freeze record also pre-registers these procedure choices:

- **Runtime:** the named profile, `profiles/flash-next-routed-f32kv-qualification`.
- **Same-input case:** the shortest long case.
- **Final norm:** the root mixer stands in for the final RMSNorm.
- **Stop tokens:** taken from the manifest's EOS list.
- **KL:** computed in float64.

## Fresh run so far

| Step | State |
| --- | --- |
| 2. Freeze inputs | Done at 16:23:01Z. 16 short cases have 16–24 tokens. The ledger cases have 2,783–2,798 tokens. The same-input case is `ledger-05`. |
| 3. CPU reference (round 0 plus 7 greedy rounds) | **Stopped.** It started at 16:23:39Z, CPU-only and niced. At 16:34:44Z the memory-pressure guard stopped it at Layer 35 of 48 in round 0: PSI `some avg10` was 2.12 with 63 GiB still available. It produced no outputs. Pressure stayed around 0.5 with the job stopped (swap full), so it was not restarted. |
| 4–7. GPU: unforced run, plain run, forced indices, same-input capture, task quality | **Not started.** The coordinator put GPU work on hold: the host's amdgpu driver is wedged. |
| 8. Evaluator | Not run. It needs steps 4–7. |

Component: `doppler.tests`, `doppler.repository-tooling`. Intent: preserved.
Boundary effects: none.
