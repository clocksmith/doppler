# Runtime Ownership Decisions

Doppler does not assume that it should own execution for every supported
workload. A governed incumbent wrapper can own a workload when it provides
equivalent correctness, usability, diagnosis, memory, performance, and artifact
control.

## Authority

`benchmarks/vendors/runtime-ownership-decisions.json` is the status authority.
`npm run runtime:ownership:check` validates its decision records and regression
tests. An empty decision registry is valid and explicitly incomplete.

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
reranking. Every claimable decision binds logical model, Doppler artifact,
Doppler execution, source provider/artifact/execution, incumbent
provider/artifact/execution, correctness class, retained evidence,
qualification date, and expiry.

Benchmark wins alone do not satisfy this contract. Missing memory, quality,
usability, diagnostic, cost, burden, or provider-risk evidence keeps the
decision non-claimable. Dispositions are `incumbent`, `doppler`, or `dual`; a
decision never silently forces an owned runtime when the incumbent remains the
better product choice.
