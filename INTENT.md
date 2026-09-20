# INTENT: Doppler

Parent: none

## Need

Developers need useful local AI as an ordinary JavaScript dependency, without a
separate Python environment or mandatory cloud inference service, and without learning
Doppler's internal release machinery to build an application.

## Target

Deliver a simple, fast, portable JavaScript inference library, supported by Forge
model preparation and signed immutable Capsules. Runtime executes the declared
implementation; the application retains publisher trust and upgrade authority.

The next product increment is the complete copy-and-run local-search application
specified in [docs/goals.md](docs/goals.md#next-product-increment-copy-and-run-local-search).
Reuse `examples/document-search/search.js`; retain `runCapability()` as a one-shot
example. Nonzero-adapter qualification and structural cleanup follow this increment
unless a concrete defect blocks it. Embeddings and reranking complete this search
increment; generation is an independently usable, optional subsequent capability.

## Invariants

- Model execution runs locally through declared WebGPU implementations; no silent
  CPU or cloud fallback. Browser and Node qualification are separate; Bun remains
  experimental. The search application must demonstrate that documents and queries
  stay local, distinguishing permitted model acquisition from input-data egress.
- Model capsules are immutable and cryptographically bound to their conversion source, tokenizer, and target plan.
- Unsupported device features, precision mismatches, or missing extensions fail closed with explicit receipts.
- Token generation exposes observable evidence (probabilities, surprisal, top candidates) without altering model weights.
- Standalone library utility remains independent of external network coordination or commercial licensing.
- Exact-configuration execution, declared support, and external adoption are
  separate conclusions. Missing customers cannot erase physical evidence;
  customers cannot substitute for it. Narrow technical support does not require
  a customer fleet, payment, or completion of the entire capability portfolio.
- The search application owns initialization, indexing, query cancellation, and
  disposal. Healthy model sessions and compatible complete indexes survive queries;
  cancellation does not publish partial indexes or stale results.
- Cancellation before submission and cancellation afterward are distinct. After
  submission, suppress successful completion without claiming GPU interruption.
  Cleanup releases owned resources without promising immediate physical reclamation;
  qualify allocation-failure and device-loss behavior, not immunity to either.
- Host prerequisites are explicit, including a Node WebGPU provider where required.
  A qualified plan is not a guarantee of available memory or cross-GPU bit identity.
- Executable identity and release history are distinct: Capsule v3 carries signed
  release events separately. Preserve checkpoints; offline use cannot discover
  unseen revocations. Embedding upgrades must detect incompatible indexes and
  offer rebuilding from preserved documents, not arbitrary state restoration.
- Downloaded or hashed bytes are not verified artifacts until size and digest
  checks succeed. Observer progress never changes verification authority.
- Reploid owns agent and peer policy and consumes the same public inference APIs
  available to standalone developers.

## Evidence

- Clean test execution via declared package verification suites (`npm test` and `npm run check:green`).
- Objective parity passing against reference model output oracles within declared tolerances.
- Goal completion matrix recorded in `src/config/goal-completion-matrix.json`.
- Clean-environment installed local-search acceptance with both real models resident,
  lifecycle regressions, offline reopening after retaining required assets, and
  network observations covering import, indexing, queries, logging, workers, and
  service workers. Local execution is not an automatic privacy guarantee; mocked UI tests and individual
  model qualification do not substitute for application acceptance. Goal wording
  alone does not update machine-reported completion or promote a release.

## Non-goals

- Application UI, document parsing, index policy, and agent orchestration inside
  the inference runtime. A complete reference application outside it is in scope.
- Hosting cloud API endpoints or multi-tenant model serving.
- Unvalidated model conversion without reproducible evaluation evidence.

## Truth

Execution proof and verifiable receipts determine capability. Benchmarks and compatibility claims are valid only when backed by reproducible local test evidence.

---

Links:
- Root strategy: [GOALS.md](GOALS.md)
- Technical charter: [CATSCAN.md](CATSCAN.md)
