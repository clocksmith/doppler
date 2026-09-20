# Doppler Goals

## Mission & Thesis

Doppler is a JavaScript library for running AI models locally through WebGPU.

**Portable local inference is ordinary software, not an opaque cloud service.**

Make AI an ordinary application dependency, without a separate Python environment
or mandatory cloud inference service. Generation, embeddings, and reranking are
independently usable, composable capabilities; Doppler handles loading, GPU execution, streaming, cancellation,
and cleanup. Browser and Node support are qualified separately; Bun remains
experimental. Host prerequisites still apply: Node may require installation of
a WebGPU provider. JavaScript/WGSL does not mean no native dependencies anywhere.

Forge, the model compiler and preparation system behind the library, prepares
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

## Next Product Increment: Copy-and-run Local Search

Deliver a complete local-search application developers can copy, run, and modify.
Prioritize it over nonzero-adapter qualification and further structural cleanup,
except for a concrete defect blocking the application. Preserve the completed
loading and resource-ownership repairs.

Build around the existing `createDocumentSearch()` in
`examples/document-search/search.js`, not a second search engine in the generic
capability example. Open embedding and reranking sessions once, index unchanged
documents once, then reuse both sessions and the index across queries. Application
code owns parsing, retrieval/index policy, presentation, and the controller lifetime;
the runtime remains an inference library. Initial completion requires embeddings
and reranking only. Generation and retrieval-augmented generation (RAG) are optional
later work; applications would own passage selection, citations, and answer evaluation.

Deliver pinned installable runtime bytes, complete model descriptors, explicit
publisher trust, accessible immutable artifacts, sample documents, and a start
command. Add honest loading/verification progress through the existing observer.
Prove repeat search, cancellation/reuse, failure cleanup, index compatibility,
and local-only document/query processing from a clean installed consumer with
both real models resident. Prove offline reopening after all required assets,
release metadata, and the index are retained. This is a required increment, not a claim of completion
or independent adoption. The detailed acceptance contract is in
[docs/goals.md](docs/goals.md#next-product-increment-copy-and-run-local-search).

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
