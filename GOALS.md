# Doppler Goals

## Mission & Thesis

Doppler is a JavaScript library for running AI models locally through WebGPU.

**Portable local inference is ordinary software, not an opaque cloud service.**

Make AI an ordinary application dependency, without a separate Python environment
or mandatory cloud inference service. Generation, embeddings, reranking, and typed choice scoring are
independently usable, composable capabilities; Doppler handles loading, GPU execution, streaming, cancellation,
and cleanup. Choice scores have an explicit model/operation identity and defined
interpretation; calibration requires separate evidence. Applications own decisions,
permissions and side effects. Browser and Node support are qualified separately; Bun remains
experimental. Host prerequisites still apply: Node may require installation of
a WebGPU provider. JavaScript/WGSL does not mean no native dependencies anywhere.

Rig, the model compiler and preparation system behind the library, prepares
supported sources, weights, and GPU programs and evaluates correctness and
performance. A Capsule is its signed, versioned executable model package; a
TargetPlan identifies a declared implementation and its execution requirements. Runtime
verifies and executes the declared implementation; applications control trusted
publishers and upgrades. Speed, simplicity, portability, and increasingly efficient
new-model support are the advantage. Verification makes that advantage dependable,
not the primary reason developers install Doppler.

## Intended Beneficiaries

1. **Client & Web Application Developers**: Engineers embedding local generation,
   reranking, and embeddings without a mandatory cloud inference service.
2. **Local Autonomous Systems (Reploid)**: Goal-directed agent runtimes requiring dependable, local token generation, embedding lookup, and verifiable execution receipts.
3. **Enterprise Privacy & Security Teams**: Organizations requiring tested local
   document/query processing and explicit control over network activity, not an
   assumption that WebGPU alone guarantees privacy.

## Desired Outcomes

1. **Independent Standalone Adoption**: Maintainers voluntarily retain Doppler-backed models because they are faster, simpler to ship, and more dependable than alternatives. Free adoption counts as full technical success.
2. **Signed Capsule Integrity**: Deliver immutable, content-addressed model packages with reproducible quantization, verified tokenizers, and execution graphs.
3. **Fail-Closed Execution**: Unsupported hardware capabilities, missing WebGPU extensions, or corrupt weights fail closed with explicit diagnostic receipts rather than falling back to unverified CPU paths.
4. **First-Class Observability**: Make loading, verification, execution, and cleanup
   understandable through honest progress and observations. Preserve token-level
   inspection and kernel timing where supported, without changing computation.

## Current priorities

This section states Doppler's public product priorities; the acceptance section
below and retained campaign evidence do not create competing work queues. The
maintainers sequence work separately; public contributors pick work from
[GitHub issues](https://github.com/clocksmith/doppler/issues) within these
boundaries.

Doppler is one independently useful JavaScript model-execution library. Prioritize
useful outputs, straightforward installation, fast opening, bounded memory,
reusable sessions, cancellation, and recovery. Generation, embeddings, reranking,
[typed choice scoring](docs/api/choice-scoring.md), and supported partition execution
use the same maintained implementation. Scoring uses contextual single-token
labels and raw logits; calibration and task quality require separate evidence.

Expand models and optimizations around demonstrated application needs. Rewriting
requires new task-quality evidence; it is not an automatic next campaign. Doe is
optional and does not replace a browser backend by download alone. Do not begin a
Doe comparison, new modality, distributed architecture, training campaign, or
model-catalog expansion merely because an older plan lists it. Verification
supports dependable execution; retained independent use establishes product value.
Numerical parity, useful answers, and ordinary application operation are separate
acceptance claims; preserve failures and unchanged references. Keep Rig → Capsule
→ Run, public compatibility contracts, explicit ownership, and
application-controlled trust and upgrades intact.

## Operating Loops

1. **Inference & Telemetry Loop**:
   ```text
   Load Capsule -> Verify Identity -> Allocate WebGPU Buffers -> Execute Kernels -> Stream Tokens & Evidence Receipts
   ```
2. **Model Qualification Loop**:
   ```text
   Source Weights -> Quantization & Graph Compilation -> Accuracy/Oracle Parity Verification -> Generate Signed Capsule & Release Receipt
   ```

## Strategic Constraints

- Provider neutrality: Doppler operates on standard WebGPU across browsers and runtimes without proprietary hardware lock-in.
- Standalone sufficiency: Doppler must succeed as a self-contained open library; payment, partner dependencies, or multi-repo integrations do not gate technical completion.
- Deterministic verification: Evidence receipts must bind exact model hashes, kernel configurations, and execution environments.
- Reploid is one consumer of the same public capabilities. It owns agents, peer
  coordination, consent, and application policy; Doppler supplies inference.
- Cancellation does not interrupt already submitted GPU commands; cleanup does
  not promise immediate physical memory reclamation. Allocation failure and device
  loss remain possible even for a qualified plan.
- Numerical acceptance is scoped to declared implementations and environments,
  not a promise of bit-identical output across every GPU. Offline operation enforces
  known release history, not unseen revocations or arbitrary cross-version state.
- Privacy is tested across the complete application. No claims of zero egress
  risk, zero infrastructure cost, instant interactions, or immunity to crashes.

## Explicit Exclusions

Doppler's runtime does not own application workflows or search/index policy.
Complete reference applications under `examples/` demonstrate ordinary public-API
consumption without adding those responsibilities to the library. Universal
unverified model catalogs, cloud-hosted API proxy layers, and unbacked performance
claims remain excluded.

## Product acceptance and evidence

This file owns Doppler's mission, priorities, and product acceptance requirements.
The technical owners below define their detailed contracts. The current joint
Doppler–Reploid integration follows Reploid's implementation roadmap, which is
maintained privately; it does not replace Doppler's independent product goals.

### Standalone adoption acceptance

Independent adoption means an unrelated application voluntarily retains Doppler
for useful outputs, simpler integration, dependable operation, or a measured
resource advantage. Internal reference integrations have separate evidence.
Neither payment nor external adoption is permission to release a technically
qualified configuration.

- [Root API](docs/api/root.md) and [getting started](docs/getting-started.md) own ordinary use.
- [Product integration qualification](docs/product-integration-qualification.md) owns
  application, workload, provider, lifecycle, and numerical identity.
- [Release contract](docs/model-release-platform.md) owns Capsule release closure
  and the non-promoting release command; the application retains activation and
  rollback authority.
- [Resident partitions](docs/distribution/resident-partition-execution.md) owns the
  public distributed execution boundary. Reploid owns placement and transport.

<a id="next-product-increment-copy-and-run-local-search"></a>

#### Search application acceptance

The historical search starter is a reference integration, not the current work
sequence. Its original signed first-install window expired; installed receipts
remain scoped to their recorded bytes and environment. Fresh distribution needs
new metadata and its own acceptance. See the
[example](examples/document-search/README.md),
[controller contract](examples/document-search/ENGINEERING.md), and
[retained maintenance receipt](artifacts/document-search-maintenance-2026-09-26/README.md).

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

### Artifact and runtime acceptance

Rig prepares source semantics and declared implementations. Capsules bind their
signed executable identity. Run validates and executes accepted compatible
implementations; applications own policy, trust, activation, and side effects.

| Subject | Contract owner |
| --- | --- |
| Artifact and catalog identity | [RDRR format](docs/rdrr-format.md), [model promotion](docs/model-promotion-playbook.md) |
| Source semantics and lowering | [ModelIR](docs/model-ir-v2-source-truth-forge.md), [conversion/runtime contract](docs/conversion-runtime-contract.md) |
| Signed releases and upgrades | [Capsule release contract](docs/model-release-platform.md#capsule-release-closure), [identity migration](docs/capsule-identity-migration.md) |
| Config and execution | [Config](docs/config.md), [pipeline contract](docs/pipeline-contract.md), [session lifecycle](docs/api/root.md) |
| Providers and hosts | [Provider conformance](docs/provider-conformance.md), [runtime ownership](docs/runtime-ownership.md), [Bun qualification](docs/bun-product-qualification.md) |
| Public command surfaces | [CLI](docs/cli.md), [tooling API](docs/api/tooling.md) |

Unsupported behavior fails explicitly. Run never silently replaces model bytes,
precision, shaders, graph meaning, or a signed TargetPlan. A simpler public API
uses this maintained engine and preserves those boundaries.

### Correctness and performance acceptance

[Benchmark methodology](docs/benchmark-methodology.md),
[claim policy](tools/policies/release-claim-policy.json), and the machine-owned support/claim registries
own promotion criteria. Freeze the correctness class, inputs, precision,
tolerance, timing boundary, controls, and regression bounds before execution.
Do not weaken acceptance after observing a failure.

Exact bits/tokens, tolerance-bounded numerics, semantic equivalence, and held-out
task quality are distinct requirements. Execution or device agreement alone
cannot substitute for the declared independent reference or application quality.
Record cold and warm costs, failed attempts, unsupported paths, and measurement
limits; a fast but incorrect or unreliable result does not qualify.

#### Bounded recursive improvement

[Optimization contracts](src/config/schema/optimization-contract.schema.json) and
[training artifact policy](docs/training-artifact-policy.md) own candidate, evaluator,
lineage, budget, and promotion boundaries. Applications retain activation
control. Training remains separately scoped; available code is not support.

### Machine reporting

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

### Historical campaigns

Earlier search, Doe comparison, network, adoption, observability, and release
sequences are historical, not additional work queues. Current artifact/API
requirements belong to the owners linked above.

---

Links:
- Strategic intent links to local invariants in [INTENT.md](INTENT.md).
- Technical boundaries and owned authority are chartered in [CATSCAN.md](CATSCAN.md).
