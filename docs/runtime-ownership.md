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

## Material advantage

Before evaluation, a decision declares at least one falsifiable hypothesis with
a metric, control metric, acceptance threshold, and declaration timestamp. A
numeric result is checked against its frozen threshold. Evaluation evidence
cannot predate the hypothesis.

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
free receipts remain retained but non-claimable. Failed external receipts are
also valid negative evidence and cannot satisfy a claimable decision.

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
