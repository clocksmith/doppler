# Provider Conformance

Provider conformance is Doppler's qualification gate for executing one exact
model/workload contract across WebGPU implementations. It does not turn a
provider probe, fixture, or Program Bundle transcript into a support claim.

## Authority

`tools/policies/provider-conformance.json` is the status authority. Its core
lanes are standard browser WebGPU and the selected Node WebGPU provider. Doe is
an optional named lane and cannot become a hidden core dependency. Other
providers must be registered explicitly before a suite may reference them.

`npm run provider:conformance:check` validates the policy and its regression
tests. An empty suite registry is structurally valid and visibly incomplete.
An exact workload tuple may be registered earlier as a non-claimable candidate
with empty provider results and explicit blockers. Candidate coverage records
what must run; it does not convert older release receipts, contract fixtures,
or desired operations into provider evidence.

## Qualification unit

One suite binds:

- a workload and workload-contract path;
- one logical model, catalog `manifestVariantId`, and resolved manifest SHA-256;
- the same declared operation set on every required provider;
- a correctness class selected before execution;
- required provider lanes and their exact execution identities.

Each provider result must identify its implementation and environment, pass
load, execute, and unload lifecycle stages, pass the suite's correctness class,
retain canonical-digest references for every required receipt, and carry current
qualification and expiry timestamps. The checker recomputes each receipt digest,
derives pass/fail from class-specific observations, and rejects an independently
authored pass flag. A required provider failure makes the suite non-claimable.

`manifestVariantId` is the catalog's stable named variant. It is not a byte
identity. `resolvedArtifactVariantId` is the `sha256:` manifest identity emitted
by runtime evidence, while `resolvedExecutionId` binds the observed provider and
execution path. The checker rejects a catalog variant name in either digest
field. A candidate may name its intended manifest variant before execution, but
its resolved digest remains null and blocked until current evidence captures it.

## Recording a provider run

`npm run provider:conformance:record -- --capture <capture.json> --out
<candidate-policy.json>` joins a retained `doppler_provider_receipt_v1` to one
declared suite and provider lane. The capture uses
`src/config/schema/provider-conformance-capture.schema.json` and supplies only
the suite, lane, dates, and retained evidence paths. The recorder derives
resolved identities from the provider receipt and derives implementation,
harness, environment, operation, lifecycle, and correctness state from the five
semantic receipts. It does not accept operator-entered outcomes or identities.

Every semantic receipt binds the exact suite, provider lane, workload, logical
model, named manifest variant, resolved artifact and execution SHA-256 values,
harness revision, environment fingerprint, and provider-receipt digest. The
model-contract receipt binds tokenizer, execution-graph, and runtime-policy
digests. Operation and lifecycle receipts expose observed work and repeated
sessions. Correctness observations are class-specific: exact-token requires
matching output digests and deterministic continuation; numerical evidence
derives its result from an explicit tolerance; semantic and held-out evidence
bind a frozen set, reference score, acceptance threshold, and maximum delta.

The default output is a separate policy for review. `--apply` is required to
write the configured policy, and `--replace` is required to replace an existing
lane result. Every recorded result remains `claimAllowed: false`, including a
fully passing run. A human provider-promotion receipt must bind the complete
provider evidence-set digest. A second human suite-promotion receipt must bind
the sorted set of required provider executions, evidence sets, promotions, and
expiry dates. Changing any retained byte or provider binding invalidates the
promotion. Recording never promotes either scope and cannot replace promoted
state. Failed semantic gates remain visible as non-claimable candidates; failed
or fallback execution receipts stay retained outside qualification state.

The accepted correctness classes are exact-token, tolerance-bounded numerical,
semantic, and held-out task metric. The suite cannot weaken this class after a
provider result is observed.

## Product gate

The initial gate requires qualified suites for generation, embedding, and
reranking across both core lanes. Adding Doe to a suite's
`requiredProviderLaneIds` is permitted only as an explicit named requirement;
it does not change the repository-wide core lanes.

Fixture-based contract tests validate structure and failure behavior only. Real
qualification requires retained browser and Node receipts with exact artifact,
execution, device, lifecycle, operation, and correctness evidence. Program
Bundle parity remains a portability diagnostic and may be linked as supporting
evidence, but it has no model-promotion or provider-support authority by itself.

Provider conformance begins at an exact tuple. Wider hardware, shape, model, or
family scope requires separate aggregate evidence; the checker never promotes a
single suite into broader authority.
