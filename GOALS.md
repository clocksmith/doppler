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

[TODO.md](TODO.md) links to the canonical implementation sequence and completion
state shared with Reploid. This section preserves product priorities; other goals
and campaign pages retain contracts and evidence without competing work queues.

Doppler is one independently useful JavaScript model-execution library. Prioritize
useful outputs, straightforward installation, fast opening, bounded memory,
reusable sessions, cancellation, and recovery. Generation, embeddings, reranking,
[typed choice scoring](docs/api/choice-scoring.md), and supported partition execution
use the same maintained implementation. Scoring uses contextual single-token
labels and raw logits; calibration and task quality require separate evidence.

1. **Finish dependable execution in the current integration.** Resolve the first
   divergent numerical operation under the frozen model, inputs, precision, and
   tolerance, then qualify the exact installed package. Exercise memory denial,
   retained weights, concurrent conversations, isolated cancellation, contributor
   loss, restart, and recovery. Reploid supplies the demanding two-computer
   workload through public interfaces; Doppler owns loading, computation, and
   execution-state lifecycle. Reploid owns discovery, placement, grants, transport,
   readiness advertisement, and conversation policy. A requester downloading no
   weights is this integration's requirement, not a restriction on standalone use.
   Numerical parity, useful answers, and ordinary application operation are
   separate acceptance claims. Preserve failures and unchanged references.
2. **Make ordinary installation and one complete integration dependable.** Ship
   accessible, pinned runtime/model bytes, a small public API, a runnable example,
   and understandable failures. Verify fresh installation, acquisition, opening,
   memory, repeated requests, cancellation, and offline reopening on each claimed
   host. The retained search starter's fresh-install eligibility has expired;
   renew its signed metadata and delivery before promoting it as current onboarding.
   Retain existing search and generation evidence without extending its scope.
3. **Earn independent application value.** Work with document-software teams that
   have concrete local-operation or deployment constraints. Compare real tasks
   against their strongest feasible alternative; measure output quality, first
   useful result, memory, acquisition, and integration effort. Freeze acceptance
   before changing code and retain unfavorable results. Independent adoption is
   a product outcome; it does not gate technical investigation or scoped release.

Expand models and optimizations around demonstrated application needs. Rewriting
requires new task-quality evidence and customer pull; it is not an automatic next
campaign. Doe is optional and does not replace a browser backend by download alone.
Do not begin a Doe comparison, new modality, distributed architecture, training
campaign, or model-catalog expansion merely because an older plan lists it.
Verification supports dependable execution; retained independent use establishes
product value. Keep Rig → Capsule → Run, public compatibility contracts, explicit
ownership, and application-controlled trust and upgrades intact.

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
- Standalone sufficiency: Doppler must succeed as a self-contained open library; revenue, partner dependencies, or multi-repo integrations do not gate technical completion.
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

---

Links:
- Strategic intent links to local invariants in [INTENT.md](INTENT.md).
- Technical boundaries and owned authority are chartered in [CATSCAN.md](CATSCAN.md).
