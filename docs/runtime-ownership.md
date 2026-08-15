# Runtime Ownership Decisions

Doppler does not assume that it should own execution for every supported
workload. A governed incumbent wrapper can own a workload when it provides
equivalent correctness, usability, diagnosis, memory, performance, and artifact
control.

## Authority

`benchmarks/vendors/runtime-ownership-decisions.json` is the status authority.
`npm run runtime:ownership:check` validates its decision records and regression
tests. An empty decision registry is valid and explicitly incomplete.
An exact portfolio tuple may be registered earlier as a non-claimable candidate
with intended controls, a frozen hypothesis, null results, and explicit
blockers. That freezes what should be tested without turning old benchmark
receipts into retrospective hypothesis evidence.

One decision compares three executions for an exact workload tuple:

1. The authoritative source execution.
2. The strongest relevant instrumented incumbent.
3. Doppler's exact resolved artifact and execution.

The incumbent must be selected for relevance, not because it is easy for
Doppler to beat. Every decision retains correctness, task quality, usability,
memory, end-to-end performance, diagnostic depth, distribution cost,
integration burden, and provider-risk evidence.
Every retained evidence reference binds a repo-relative path and the canonical
JSON SHA-256 of that receipt. Replacing bytes behind a path invalidates the
decision even when the replacement remains valid JSON.

## Material advantage

Before evaluation, a decision declares at least one falsifiable hypothesis with
a metric, control metric, acceptance threshold, and declaration timestamp. A
numeric result is derived from a semantic hypothesis receipt and checked against
its frozen threshold. That receipt also binds the exact source, incumbent, and
Doppler executions, harness revision, environment fingerprint, control result,
and end-to-end acceptance result. Operators cannot author the retained observed
value or pass/fail result independently. Evaluation evidence cannot predate the
hypothesis.

Recognized advantage axes are unsupported operation, end-to-end performance,
memory, diagnostic depth, offline artifact control, and verified correction
path. A `doppler` or `dual` disposition requires at least one passing material
advantage. An `incumbent` disposition is invalid if such an advantage passed.

## Product gate

The initial gate requires current decisions for generation, embedding, and
reranking. Every claimable decision binds logical model, named Doppler manifest
variant, resolved Doppler manifest SHA-256, Doppler execution SHA-256, source
provider/artifact/execution, incumbent
provider/artifact/execution, correctness class, retained evidence,
qualification date, and expiry.

The catalog `manifestVariantId` and runtime `resolvedArtifactVariantId` are not
interchangeable. The former identifies the intended named artifact lane; the
latter is the immutable `sha256:` manifest identity observed during execution.
The checker rejects catalog variant names in resolved artifact or execution
digest fields, so a pre-execution candidate must leave those fields null rather
than imply byte identity.

Source and incumbent executions use
`doppler.runtime-ownership-execution-evidence/v1`, defined by
`benchmarks/vendors/schema/runtime-ownership-execution-evidence.schema.json`.
The receipt binds provider, artifact revision, logical model, workload, runtime
and backend versions, environment, invocation configuration, output, status,
and timestamps. `sourceExecutionId` and `incumbentExecutionId` are the canonical
JSON SHA-256 identities of those receipts. Generate an ID with `npm run
runtime:ownership:evidence-id -- --receipt <receipt.json>`; the ownership checker
recomputes it and rejects tampered evidence or mismatched provider, artifact,
model, role, or workload fields.

Doppler does not substitute a receipt digest for its native execution identity.
Its `dopplerExecution` evidence must be a local
`doppler_provider_receipt_v1` whose resolved manifest and execution SHA-256
values exactly match the decision. Fallback, failed, unresolved, or environment-
free receipts remain retained but non-claimable. Failed source receipts remain
negative evidence. A failed incumbent receipt can support a claim only when a
predeclared unsupported-operation hypothesis passes and the correctness receipt
independently records acceptable source and Doppler behavior plus unacceptable
incumbent behavior; otherwise the failure remains disqualifying.

The nine non-execution dimensions use
`doppler.runtime-ownership-dimension-evidence/v1`, defined by
`benchmarks/vendors/schema/runtime-ownership-dimension-evidence.schema.json`.
Each receipt names one evidence class, binds the same immutable execution
identities, records a harness revision and environment fingerprint, and carries
class-specific observations. The checker recomputes its result: quality binds a
held-out set and acceptance score; memory and distribution cost use measured
byte budgets; performance requires matched work, timing scope, and sufficient
samples; integration uses bounded steps plus clean-install and API success; and
provider risk requires standard WebGPU and the selected Node provider without
an undeclared Doe dependency. A generic file or independently authored
`passed` value cannot satisfy a dimension.

Every dimension and hypothesis in one decision must bind the same harness
revision and environment fingerprint. Those identities are stored on the
decision and checked across all semantic receipts, so results from unrelated
evaluation runs cannot be composed into one disposition.

Material-advantage results use
`doppler.runtime-ownership-hypothesis-evidence/v1`, defined by
`benchmarks/vendors/schema/runtime-ownership-hypothesis-evidence.schema.json`.
The checker derives threshold success only when both the frozen control and the
end-to-end acceptance gate pass. Failed receipts remain retained negative
evidence; they do not become a material advantage.

## Recording an evaluation

`npm run runtime:ownership:record -- --capture <capture.json> --out
<candidate-policy.json>` joins the three execution receipts, every required
decision-evidence path, and results for the already declared hypotheses. The
capture contract is
`benchmarks/vendors/schema/runtime-ownership-evaluation-capture.schema.json`.
It cannot provide execution IDs or change a hypothesis statement, metric,
control, threshold, declaration time, observed value, pass/fail result, or
evidence digest. The recorder reads paths, derives canonical digests and external
receipt identities, reads Doppler's native identities, validates every semantic
dimension, derives hypothesis results, and rejects missing axes, retrospective
evidence, false result claims, reused evidence paths, and inconsistent
disposition recommendations.

The default output is a separate policy for review. `--apply` is required to
write the configured policy, and `--replace` is required when a candidate
already contains evaluation state. A claimable decision cannot be replaced.
Every recorded evaluation remains `claimAllowed: false` with an explicit
promotion blocker; failed executions remain additional blockers. Promotion is
a separate human decision after reviewing the complete comparison. The recorder
clears promotion evidence and cannot replace a promoted disposition.

`promotion` uses `doppler.runtime-ownership-promotion-evidence/v1`, defined by
`benchmarks/vendors/schema/runtime-ownership-promotion-evidence.schema.json`.
A human reviewer binds the exact source, incumbent, Doppler, harness,
environment, correctness-class, disposition, and qualification identities plus
canonical digests of every comparison reference and the complete hypothesis
set. The decision must be `promote-disposition`. Editing `claimAllowed`,
blockers, or a disposition cannot publish a runtime-ownership claim without
this receipt, and changing any bound evidence or hypothesis invalidates prior
promotion.

Benchmark wins alone do not satisfy this contract. Missing memory, quality,
usability, diagnostic, cost, burden, or provider-risk evidence keeps the
decision non-claimable. Dispositions are `incumbent`, `doppler`, or `dual`; a
decision never silently forces an owned runtime when the incumbent remains the
better product choice.

The current candidates deliberately reuse the generation, embedding, and
reranking portfolio selected by the integration and provider gates. Historical
comparison receipts may guide execution planning, but they predate the frozen
ownership hypotheses and do not cover the complete source/incumbent/Doppler
decision contract. A failed material-advantage hypothesis is a valid result and
should lead to an `incumbent` disposition when no other predeclared advantage
passes.
