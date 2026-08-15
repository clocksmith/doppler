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
retain every required evidence path, and carry current qualification and expiry
timestamps. A required provider failure makes the suite non-claimable.

`manifestVariantId` is the catalog's stable named variant. It is not a byte
identity. `resolvedArtifactVariantId` is the `sha256:` manifest identity emitted
by runtime evidence, while `resolvedExecutionId` binds the observed provider and
execution path. The checker rejects a catalog variant name in either digest
field. A candidate may name its intended manifest variant before execution, but
its resolved digest remains null and blocked until current evidence captures it.

The accepted correctness classes are exact-token, tolerance-bounded numerical,
semantic, and held-out task metric. The suite cannot weaken this class after a
provider result is observed.

## Product gate

The initial gate requires qualified suites for generation, embedding, and
reranking across both core lanes. Adding Doe to a suite's
`requiredProviderLaneIds` is permitted only as an explicit named requirement;
it does not change the repository-wide core lanes.

Fixture-based contract tests validate failure behavior only. Real qualification
requires retained browser and Node receipts with exact artifact, execution,
device, lifecycle, operation, and correctness evidence. Program Bundle parity
remains a portability diagnostic and may be linked as supporting evidence, but
it has no model-promotion or provider-support authority by itself.

Provider conformance begins at an exact tuple. Wider hardware, shape, model, or
family scope requires separate aggregate evidence; the checker never promotes a
single suite into broader authority.
