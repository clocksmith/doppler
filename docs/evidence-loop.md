# Model and Kernel Evidence Loop

Doppler treats “optimal” as a digest-bound, evidence-backed selection for one
artifact, execution graph, browser, adapter capability fingerprint, phase, and
shape class. It does not mean one universal best kernel.

This workflow extends existing manifests, operator diagnostics, benchmark
receipts, and runtime optimization contracts. It does not create a second
execution architecture.

## 1. Inspect Source Material Before Conversion

Run:

```bash
doppler onboard inspect \
  --source /path/to/source-checkpoint \
  --out artifacts/onboarding/model-id
```

The command reads `config.json`, optional generation and tokenizer configs, and
SafeTensors headers without reading tensor payloads. It writes:

- `source-intake.json` using `doppler.source-intake/v1`
- `conversion-config.skeleton.json`
- `contract-tests.plan.json`

Every fact records its source file and JSON pointer, Doppler owner, proposal,
confidence, status, and verification method. Confidence is one of `direct`,
`derived`, `family-inferred`, `ambiguous`, or `unsupported`. Only accepted
direct or derived facts enter the skeleton. Neighboring-family values remain
unresolved proposals:

```bash
doppler onboard inspect \
  --source /path/to/variant \
  --family-intake artifacts/onboarding/family/source-intake.json \
  --out artifacts/onboarding/variant
```

This command is pre-conversion. The existing `doppler intake` command remains
the post-conversion manifest and execution-contract check.

## 2. Compare Semantic Boundaries

Source runtimes first emit `doppler.boundary-provider-capture/v1`. Bind that
provider output into the sole source-pack contract:

```bash
doppler boundary source-pack \
  --provider-capture artifacts/provider-boundaries.json \
  --out artifacts/source-boundaries.json
```

The resulting `doppler.source-boundary-pack/v1` includes:

- semantic boundary ID
- shape and dtype
- deterministic sample coordinates and values
- full-tensor SHA-256 digest
- numerical statistics
- tolerance-policy ID

Boundary IDs describe model semantics, for example:

```text
layer.0.attention.q.pre_rope
layer.0.attention.q.post_rope
layer.0.attention.output
layer.0.ffn.output
model.logits
```

Capture a Doppler runtime report produced with operator diagnostics:

```bash
doppler boundary capture \
  --report artifacts/diagnose-report.json \
  --out artifacts/runtime-boundaries.json
```

Compare it with the source pack:

```bash
doppler boundary token-evidence \
  --reference-transcript artifacts/reference-transcript.json \
  --out artifacts/greedy-128-token-evidence.json

doppler boundary compare \
  --source-pack artifacts/source-boundaries.json \
  --runtime-capture artifacts/runtime-boundaries.json \
  --token-evidence artifacts/greedy-128-token-evidence.json \
  --out artifacts/boundary-comparison.json
```

The comparison receipt uses `doppler.boundary-comparison-receipt/v1` and stops
at the first divergent semantic boundary. Quantized comparisons additionally
require `--artifact-precision quantized --source-control <f16-receipt>`.
Promotion requires compatible boundaries and exact deterministic parity for at
least 128 generated tokens.

## 3. Emit a Calibrate-Safe Token Cost Ledger

Enable the standard execution observer:

```json
{
  "shared": {
    "benchmark": {
      "run": {
        "executionObserver": {
          "enabled": true,
          "includeDispatchGeometry": true,
          "estimateBytesMoved": false
        }
      }
    }
  }
}
```

Both `shared.benchmark.run.executionObserver` and `shared.debug.profiler`
activate the same `CommandRecorder` observation path and emit the same
`doppler.token-cost-ledger/v1` at `result.metrics.tokenCostLedger`. Calibrate
intent uses the standard execution-observer flag because debug profiling is
intentionally forbidden there; debug intent may use the profiler flag. There
is no second profiler implementation.

The ledger separates prefill and decode. It records timestamp or CPU-estimated
measurement source, attributed and unattributed time, timestamp coverage,
dispatches, known workgroups, command-buffer submissions, observer readbacks,
selected execution information, digest identities, command-recording time, and
submit/readback fence observations. `submitWaitMs` and `readbackWaitMs` may
observe the same GPU work, so `fenceWaitMs` is their maximum rather than their
sum. Map, cleanup, and copy measurements are a nested readback breakdown and
are not added again. Bytes moved are labeled estimated and remain unavailable
unless an estimator supplies them. Summed GPU operation durations are never
asserted to equal wall time.

For browser inference, `artifactDigest` binds the declared artifact identity,
ordered shard hashes, tokenizer contract, and total size. `wrapperDigest` binds
the versioned text-execution wrapper contract, resolved execution plan, kernel
path, phase operation maps, and execution policies. It is distinct from
`kernelSetDigest`, which binds the registered WGSL entries.

The classifier policy in
`src/config/evidence/token-cost-classifier-policy.json` maps operation labels to
projection, attention, sampling, and memory walls. Every optimization
hypothesis should name the ledger entry it expects to reduce. The classifier
also compares those GPU walls with built-in command-recording,
submit/readback-fence, and host-orchestration walls, then returns experiments
appropriate to the largest observed cost.

## 4. Calibrate Registered Variants

First materialize the routing inventory:

```bash
npm run routing:audit
npm run routing:audit:check
```

The generated
`benchmarks/kernels/execution-routing-audit.json` verifies every pinned kernel
digest it can resolve and lists exact-head prefill, tiled prefill, and F16-output
alternatives. Every alternative is marked `calibration-required`; the audit
does not rewrite a manifest or runtime profile.

`calibrateRegisteredVariants()` consumes:

- resolved artifact, manifest, and execution-graph digests
- browser and adapter digests plus capability set
- baseline and candidate references from the kernel registry
- complete shape signatures
- callbacks that run actual operator, boundary, token, and performance checks

A shape signature includes phase, sequence length, batch, query/KV head
geometry, tail class, layouts, storage/materialization/accumulation dtypes,
fusion role, and quantization format.

The synthetic runtime tuner has been removed. Registered execution variants
running their real WGSL and wrappers are the only kernel-selection candidates.

Registered calibration gates candidates in this order:

1. operator/reference correctness for every mandatory shape
2. semantic boundary-pack compatibility, including the source-precision control
3. exact deterministic parity for at least 128 tokens
4. end-to-end performance through the runtime optimization evaluator

Successful calibration emits a proposed runtime profile or registered
execution-graph patch. It never activates a runtime mutation.

F16 selection follows one strict rule: hardware support makes an F16 candidate
eligible, while source-bound accuracy and positive paired performance make it
mandatory. After promotion, the F16 route is the default for its bound adapter,
execution engine, artifact, execution graph, phase, and shape scope. F32 remains legal
only when shader-f16 is unavailable, a declared stable boundary requires F32,
the F16 result is not faster, or its evidence has been revoked. Capability
alone never promotes an unverified precision change.

Run a digest-bound job through the actual command surface:

```bash
npm run calibrate:registered -- \
  --job artifacts/calibration/job.json \
  --out artifacts/calibration/receipt.json

bun tools/calibrate-registered-variants.js \
  --job artifacts/calibration/job.json \
  --out artifacts/calibration/bun-receipt.json
```

Set `surface` in the job to `node` for the Node/Bun WebGPU command runner or
`browser` for the Playwright Chromium relay. A job embeds the calibration plan,
correctness-evidence bindings, one checked-in candidate registry, one runtime
optimization contract per candidate, and optional command-runner settings.
Each correctness binding must repeat the artifact, execution-graph, descriptor,
kernel, execution-engine, browser, and adapter digests. The candidate registry
evidence scope must repeat the same seven values. The job also checks candidate
and baseline kernel digests against
the current WGSL digest registry. `executionEngineDigest` is the canonical hash
of the active surface and engine string; every benchmark result must also match
the plan's artifact, execution graph, adapter, and browser digest through its
token-cost-ledger identity. Missing, historical, or mismatched evidence fails
before benchmarking; historical manifest pins remain visible in the routing
audit until those artifacts are requalified rather than silently re-signed.

The command executes verify, interleaved paired measurement, and declared
neighboring-workload guards through Doppler's normal command envelopes. Its
receipt records the execution surface, concrete Node/Bun/browser engine, and
job digest. There is no fixture-only success path and no generated hardware
result in CI.

## 5. Promote Through the Existing Evaluator

Every runtime optimization contract freezes its campaign guardrails before a
candidate runs. The required `campaign` object names an owner, change class,
falsifiable causal hypothesis, expected metric, unchanged control, end-to-end
acceptance metric, candidate and command-run budgets, stopping rule, retry
conditions, and revocation conditions. The validator cross-checks those fields
against executable verification, measurement, mutation, neighboring-workload,
and sequential-decision policy; duplicated prose cannot quietly disagree with
the run.

The evaluator copies the frozen campaign into every accepted, rejected, or
invalid receipt. It can recommend a candidate, but it never mutates production:
promotion authority remains human and requires shadow and canary stages. The
receipt index retains negative results so a rejected candidate is not retried
without satisfying its named retry conditions.

Human promotion starts a separate post-promotion contract in
`tools/policies/runtime-promotion-monitoring.json`. The plan must be declared
before activation and binds the accepted optimization receipt, exact model,
artifact, execution, provider, environment, workload, primary metric,
unchanged controls, neighboring workloads, observation count, known-safe
rollback target, and original revocation conditions. Each observation repeats
the same scope inside a
`doppler.runtime-promotion-observation-evidence/v1` JSON receipt. The policy
binds every observation, rollback proof, and terminal-decision proof by its
canonical JSON SHA-256, so retained values and pass flags cannot drift behind
unchanged paths.

Actual activation uses
`doppler.runtime-promotion-activation-evidence/v1`; terminal retain or revoke
uses `doppler.runtime-promotion-decision-evidence/v1`. Both name a human
reviewer revision and bind the candidate hash and exact execution scope.
Automation cannot substitute its own activation assertion or terminal outcome.

`npm run promotion:monitoring:record -- --capture <capture.json> --out
<candidate-policy.json>` materializes the monitoring record from an accepted
optimization receipt, activation receipt, known-safe rollback evidence,
observation receipt paths, and optional terminal-decision receipt. It derives
candidate facts, hashes, scope, revocation conditions, observation digests, and
decision fields. The default output is separate; `--apply` is explicit and
`--replace` protects existing records. The recorder writes
`runtimeMutationApplied: false` and never performs activation, rollback, or
revocation itself. Replacement is monotonic: frozen activation, scope, plan,
rollback, and accepted-candidate facts cannot change; prior observations cannot
be removed or rewritten; and terminal decisions are immutable.

`npm run promotion:monitoring:check` recomputes the decision. A frozen primary-
metric degradation or any failed control or neighbor requires `revoke`; enough
passing observations require `retain`; otherwise the record remains
`monitoring`. The evaluator never mutates production. A revoke decision is
valid only when the bundled revocation registry contains a matching active
record. The current registry has no promoted candidates, so exercised coverage
remains explicitly incomplete.

`doppler.runtime-optimization-contract/v1` supports:

- `runtime-profile` (`runtime_profile` remains a compatibility spelling)
- `registered-kernel-variant`
- `registered-execution-graph-patch`

Ordinary runtime-profile candidates retain the runtime-owned mutation allowlist.
Registered kernel and graph candidates contain no inline patch. They reference a
digest-bound entry under
`src/config/runtime/optimization-candidates/`, and the evaluator verifies the
entry payload digest before materialization.

Measurement can use balanced randomized blocks. Optional sequential decisions
use predeclared Bonferroni fixed looks, preventing ordinary confidence-interval
peeking. Neighboring-workload guards run parity and paired regression checks
before acceptance.

Build the queryable accepted/rejected evidence index from saved receipts:

```bash
npm run optimization:index -- \
  --receipts artifacts/optimization-receipts \
  --out artifacts/runtime-optimization-results-index.json
```

The index is derived evidence. Do not hand-edit it.

Run `npm run evidence:workflows:check` to verify that each job above still has
one canonical owner and that retired synthetic-selection, duplicate-profiler,
and standalone-receipt surfaces have not returned.

The same ownership registry covers product qualification after measurement.
Maintained applications, provider conformance, runtime ownership, Bun product
support, post-promotion monitoring, and signed revocation authority each have
one non-promoting recorder and one validating check. Their receipts may reuse an
exact execution only when every gate's semantic binding agrees; no cross-gate
wrapper may invent results, weaken a gate, or silently promote production.

## 6. Keep the Runtime Path Cheap

Ordinary generation does not retain per-dispatch geometry or label histograms.
Those objects are enabled only for the execution observer, benchmark, or debug
inspection. Layer-pipeline phase selection is compiled once into immutable
per-layer prefill/decode step arrays; token execution consumes those arrays
without rescanning overrides or allocating filtered plans.

Canonical rule JSON remains split by policy owner. Runtime loads the generated
`src/rules/generated/rule-bundle.json` so a browser does not fetch every rule
file independently. `npm run rules:bundle:check` proves that mirror is current.
