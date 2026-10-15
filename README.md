# Doppler

A JavaScript library for local model execution through WebGPU: generation,
embeddings, reranking, and choice scoring, qualified per model and host.

**[Try the demo](https://d4da.com/doppler/)** · [Get started](docs/getting-started.md) · [Documentation](docs/INDEX.md)

<picture>
  <source media="(max-width: 640px)" srcset="assets/readme/application-mobile.svg" />
  <img src="assets/readme/application.svg" alt="Rig prepares a Capsule; an application reuses loaded Doppler weights and programs across requests." />
</picture>

## Get started

Run the pinned published quickstart with Node 20+ and a working WebGPU provider:

```sh
npx --package doppler-gpu@0.6.1 doppler-gpu --model gemma3-270m --prompt "Describe WebGPU briefly"
```

First execution downloads model assets. Browser and Node support are qualified
separately. The [installation guide](docs/getting-started.md) covers provider
requirements and embedding the published compatibility API.

The checkout's [signed Run API](docs/api/root.md) and resident partition exports
have separate release requirements; do not assume they exist in npm 0.6.1.

## Technical architecture

### Preparation, host, and execution ownership

![Purple session boundaries distinguish signed Capsule opening, persistent model resources, and successive operations through a model program, binder, and executor. The application and WebGPU provider remain external.](assets/readme/technical-architecture.svg)

Rig prepares signed Capsules with declared TargetPlans. Run verifies artifacts
and selects an accepted plan; session operations reuse resident resources through
the model program, binder, and command executor. Applications supply trust and
host ports. JSON declares execution, JavaScript orchestrates, and WGSL computes.
Doe and Reploid remain optional integrations.

[Architecture and source owners](docs/architecture.md) · [Resident partition lifetimes](docs/architecture.md#resident-partitions-and-resource-lifetime)

## Evidence

### Selected wins against Transformers.js

**18.8× faster warm generation on Qwen 3.5 0.8B; 16.3× on Qwen 3.5 2B.**
These retained July 2026 Radeon 8060S browser runs used 512 prompt tokens and
128 generated tokens on a synthetic raw-text workload, with exact output matches.
The Apple M3 0.8B run was 4.1×
faster. Generation includes prompt processing and decode; model load is separate.

<picture>
  <source media="(max-width: 640px)" srcset="assets/readme/generation-time-mobile.svg" />
  <img src="assets/readme/generation-time.svg" alt="Selected July 2026 warm-generation times: Qwen 0.8B Radeon, Doppler 2.80s vs Transformers.js 52.66s; Qwen 2B Radeon, 3.45s vs 56.38s; Qwen 0.8B Apple M3, 5.44s vs 22.40s. Loading is shown separately in the chart." />
</picture>

These are **RDRR Q4K vs ONNX q4f16 product-format comparisons** against
Transformers.js 4.0.1, not identical-format kernel benchmarks or measurements of
today's releases. The 2B Radeon result uses a local artifact. Each card links its
receipt; [the scoreboard](docs/model-competition-scoreboard.md) includes decode,
retrieval, correctness and missing-evidence details. Selected wins do not imply
every model, device, load phase or task is faster.

**JavaScript orchestration. WGSL compute. No mandatory ONNX Runtime dependency.**
This checkout declares zero mandatory npm runtime dependencies; model assets and
the host's WebGPU provider are separate. Rig prepares model programs; Run executes
them and reuses resident resources across requests. See the
[source ownership map](docs/architecture.md). This is an architecture/dependency
statement, not a measured browser-bundle-size or lines-of-code comparison.

<!-- model-type-clusters:start -->

**Model catalog:** text generators, multimodal generators, diffusion language models, translation specialists, language embedders, rerankers, protein encoders, nucleotide encoders.
The [support matrix](docs/model-support-matrix.md) separates catalog entries
from verified execution and links the lifecycle evidence for each model.

<!-- model-type-clusters:end -->

The [comparison scoreboard](docs/model-competition-scoreboard.md) reports
throughput and loading separately against declared controls. Runtime correctness,
task quality, memory, and lifecycle acceptance remain separate checks.
W8A8 remains experimental; it does not qualify distributed chat.

Choice scores are uncalibrated. Unsupported paths fail closed. The retained
[search starter](examples/document-search/README.md) needs renewed signed
eligibility before fresh-install use; see [installation limits](docs/getting-started.md#retained-search-starter).

## Read next

- [API index](docs/api/index.md) — operations and public entrypoints.
- [Architecture](docs/architecture.md) — preparation and execution boundaries.
- [Goals](GOALS.md) · [Contributing](docs/contributing.md) — priorities and development.

## License

[MIT](LICENSE). See [NOTICE](NOTICE) for attribution.
