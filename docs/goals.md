# Product acceptance and evidence

[GOALS.md](../GOALS.md) owns Doppler's mission and current priorities.
This page routes acceptance to its technical owners. The current joint
Doppler–Reploid integration uses Reploid's
[implementation checklist](https://github.com/clocksmith/reploid/blob/main/TODO.md);
it does not replace Doppler's independent product goals.

## Goal 1: Earn standalone executable-model adoption

Independent adoption means an unrelated application voluntarily retains Doppler
for useful outputs, simpler integration, dependable operation, or a measured
resource advantage. Internal reference integrations and paid release operations
have separate evidence. Neither payment nor external adoption is permission to
release a technically qualified configuration.

- [Root API](api/root.md) and [getting started](getting-started.md) own ordinary use.
- [Product integration qualification](product-integration-qualification.md) owns
  application, workload, provider, lifecycle, and numerical identity.
- [Release operations](model-release-platform.md) owns optional adoption and
  release acceptance; the application retains activation and rollback authority.
- [Resident partitions](distribution/resident-partition-execution.md) owns the
  public distributed execution boundary. Reploid owns placement and transport.

<a id="next-product-increment-copy-and-run-local-search"></a>

### Search application acceptance

The historical search starter is a reference integration, not the current work
sequence. Its original signed first-install window expired; installed receipts
remain scoped to their recorded bytes and environment. Fresh distribution needs
new metadata and its own acceptance. See the
[example](../examples/document-search/README.md),
[controller contract](../examples/document-search/ENGINEERING.md), and
[retained maintenance receipt](../artifacts/document-search-maintenance-2026-09-26/README.md).

The retained acceptance requirements are:

- Install public exports outside the checkout without private imports or
  accidental ancestor dependency resolution.
- Keep embedding and reranking sessions resident together; reuse unchanged
  documents and compatible indexes across repeated queries.
- Coalesce initialization, close partially initialized sessions, preserve the
  working index when replacement fails, suppress superseded results, and settle
  disposal during loading or execution.
- Cancel a query and successfully reuse healthy sessions. Submitted GPU work
  retains resources until its completion owner settles it.
- Check real-model results against the frozen reference with both models
  resident, including compatible index reuse/rejection and offline reopening.
- Observe application requests and logging to verify that documents and queries
  remain local; permitted model acquisition does not prove private-input egress.
- Measure installation, opening, useful results, repeated queries, memory,
  cancellation, and cleanup separately. Distinguish mocks from physical runs.
- Bind installed archive, model, tokenizer, implementation, provider, surface,
  hardware, and driver identities. Browser, Node, Electron, and Bun require
  separate qualification; storage eviction and unavailable memory remain possible.

These requirements do not authorize another search, model, or provider campaign.

## Goal 2: Own the model artifact and runtime contract

Rig prepares source semantics and declared implementations. Capsules bind their
signed executable identity. Run validates and executes accepted compatible
implementations; applications own policy, trust, activation, and side effects.

| Subject | Contract owner |
| --- | --- |
| Artifact and catalog identity | [RDRR format](rdrr-format.md), [model promotion](model-promotion-playbook.md) |
| Source semantics and lowering | [ModelIR](model-ir-v2-source-truth-forge.md), [conversion/runtime contract](conversion-runtime-contract.md) |
| Signed releases and upgrades | [Capsule release contract](model-release-platform.md#capsule-release-closure), [identity migration](capsule-identity-migration.md) |
| Config and execution | [Config](config.md), [pipeline contract](pipeline-contract.md), [session lifecycle](api/root.md) |
| Providers and hosts | [Provider conformance](provider-conformance.md), [runtime ownership](runtime-ownership.md), [Bun qualification](bun-product-qualification.md) |
| Public command surfaces | [CLI](cli.md), [tooling API](api/tooling.md) |

Unsupported behavior fails explicitly. Run never silently replaces model bytes,
precision, shaders, graph meaning, or a signed TargetPlan. A simpler public API
uses this maintained engine and preserves those boundaries.

## Goal 3: Make correctness and performance evidence-backed

[Benchmark methodology](benchmark-methodology.md),
[claim policy](../tools/policies/release-claim-policy.json), and the machine-owned support/claim registries
own promotion criteria. Freeze the correctness class, inputs, precision,
tolerance, timing boundary, controls, and regression bounds before execution.
Do not weaken acceptance after observing a failure.

Exact bits/tokens, tolerance-bounded numerics, semantic equivalence, and held-out
task quality are distinct requirements. Execution or device agreement alone
cannot substitute for the declared independent reference or application quality.
Record cold and warm costs, failed attempts, unsupported paths, and measurement
limits; a fast but incorrect or unreliable result does not qualify.

### Bounded recursive improvement

[Optimization contracts](../src/config/schema/optimization-contract.schema.json) and
[training artifact policy](training-artifact-policy.md) own candidate, evaluator,
lineage, budget, and promotion boundaries. Applications retain activation
control. Training remains separately scoped; available code is not support.

## Machine reporting

Registries and retained receipts own status; prose does not copy their verdicts.

| Command | Authority |
| --- | --- |
| `npm run goals:check` / `goals:report` | `src/config/goal-completion-matrix.json` |
| `npm run product:readiness:report` / `product:readiness:check` | Exact-archive acceptance, support, adoption, and deployment projections |
| `npm run product:integrations:check` | `tools/policies/product-integration-qualification.json` |
| `npm run provider:conformance:check` | `tools/policies/provider-conformance.json` |
| `npm run runtime:ownership:check` | `benchmarks/vendors/runtime-ownership-decisions.json` |
| `npm run revocations:check` / `revocations:authority:check` | Bundled deny records and separately qualified signed authority |
| `npm run promotion:monitoring:check` | `tools/policies/runtime-promotion-monitoring.json` |
| `npm run claims:evidence:check` | `tools/policies/claim-evidence-contract.json` |
| `npm run artifact:contract:check` | `tools/policies/model-artifact-contract.json` |
| `npm run commands:surface:check` | `tools/policies/command-surface-contract.json` |
| `npm run policy:schemas:check` | `src/config/schema/policy-schema-registry.json` |
| `npm run support:subsystems:check` | `tools/policies/subsystem-support-contract.json` |
| `npm run catscan:check` | Component charters and their generated index |

Readiness binds the selected acceptance record and archive hash, not HEAD or a
matching version number. Technical configuration, support, adoption, deployment,
and revocation authority remain separate. Recording evidence never activates a
candidate or grants production authority. Missing evidence stays incomplete.

## Historical campaigns

The [previous goals and campaign record](https://github.com/clocksmith/doppler/blob/e0b217e646eb0da5bc83f4ca7ec18932cbe6c086/docs/goals.md) retains the original search,
Doe comparison, network, adoption, observability, and release sequences with
their evidence. Those sequences are historical, not additional work queues.
Current artifact/API requirements belong to the owners linked above.
