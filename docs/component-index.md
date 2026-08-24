# Doppler Component Index

Generated from the repository's `CATSCAN.md` files.
Run `npm run catscan:sync` after adding, removing, or changing a component charter.

Components: 29

| Component | Target | Charter | Parent |
| --- | --- | --- | --- |
| `doppler` | Deliver inspectable, evidence-backed local model execution for JavaScript applications. | [CATSCAN.md](../CATSCAN.md) | none |
| `doppler.benchmarks` | Produce reproducible measurements whose workload, identity, timing scope, and claim status are explicit. | [benchmarks/CATSCAN.md](../benchmarks/CATSCAN.md) | `doppler` |
| `doppler.demo` | Give users one dependable browser surface for verified local model execution and inspectable evidence. | [demo/CATSCAN.md](../demo/CATSCAN.md) | `doppler` |
| `doppler.docs` | Preserve navigable human contracts without competing with machine-readable status or implementation evidence. | [docs/CATSCAN.md](../docs/CATSCAN.md) | `doppler` |
| `doppler.model-catalog` | Govern the exact model identities and lifecycle facts that Doppler can resolve, qualify, and expose. | [models/CATSCAN.md](../models/CATSCAN.md) | `doppler` |
| `doppler.runtime-source` | Implement Doppler's shipped JavaScript and WGSL contracts with explicit policy resolution and deterministic ownership. | [src/CATSCAN.md](../src/CATSCAN.md) | `doppler` |
| `doppler.tests` | Provide executable evidence that contracts hold on success, failure, lifecycle, parity, and regression paths. | [tests/CATSCAN.md](../tests/CATSCAN.md) | `doppler` |
| `doppler.repository-tooling` | Automate repository governance, generation, qualification, and operator workflows without entering the shipped runtime. | [tools/CATSCAN.md](../tools/CATSCAN.md) | `doppler` |
| `doppler.benchmarks.vendors` | Compare Doppler and eligible incumbent runtimes under shared, identity-bound evidence contracts. | [benchmarks/vendors/CATSCAN.md](../benchmarks/vendors/CATSCAN.md) | `doppler.benchmarks` |
| `doppler.runtime-source.cli` | Present the shared Doppler command contract as a clear, scriptable Node command-line interface. | [src/cli/CATSCAN.md](../src/cli/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.client` | Make qualified local model workloads easy to invoke while preserving exact resolved execution identity. | [src/client/CATSCAN.md](../src/client/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.config` | Resolve all runtime-visible policy from validated, traceable, single-source configuration contracts. | [src/config/CATSCAN.md](../src/config/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.converter` | Materialize source checkpoints into explicit, reproducible runtime artifacts without inventing runtime policy later. | [src/converter/CATSCAN.md](../src/converter/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.debug` | Expose structured runtime observation without silently changing execution or evidence meaning. | [src/debug/CATSCAN.md](../src/debug/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.experimental` | Contain non-core capabilities so they can evolve without being mistaken for product-supported runtime behavior. | [src/experimental/CATSCAN.md](../src/experimental/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.formats` | Parse and validate external and native artifact formats into explicit data contracts without executing models. | [src/formats/CATSCAN.md](../src/formats/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.gpu` | Execute fully resolved tensor operations on WebGPU with deterministic resource and submission ownership. | [src/gpu/CATSCAN.md](../src/gpu/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.inference` | Turn validated model and runtime contracts into deterministic semantic plans, execution, and model outputs. | [src/inference/CATSCAN.md](../src/inference/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.loader` | Materialize validated artifact tensors into runtime-owned CPU and GPU representations with bounded memory behavior. | [src/loader/CATSCAN.md](../src/loader/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.memory` | Provide explicit, reusable GPU and host-memory ownership mechanisms with measurable limits and cleanup. | [src/memory/CATSCAN.md](../src/memory/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.model-contracts` | Expose narrow model-family source contracts without reintroducing runtime family detection or catalog authority. | [src/models/CATSCAN.md](../src/models/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.rules` | Resolve explicit data-only rule maps deterministically and fail when no permitted selection exists. | [src/rules/CATSCAN.md](../src/rules/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.storage` | Persist, retrieve, verify, and transport artifact bytes through explicit storage contexts and integrity contracts. | [src/storage/CATSCAN.md](../src/storage/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.tooling` | Give browser, Node, and CLI adapters one normalized command contract and one evidence-aware execution vocabulary. | [src/tooling/CATSCAN.md](../src/tooling/CATSCAN.md) | `doppler.runtime-source` |
| `doppler.runtime-source.client.electron` | Bind an Electron renderer to an eligible immutable Pack while keeping durable update state and customer activation authority in the main process. | [src/client/electron/CATSCAN.md](../src/client/electron/CATSCAN.md) | `doppler.runtime-source.client` |
| `doppler.runtime-source.formats.rdrr` | Define and validate Doppler's canonical manifest-first runtime artifact contract. | [src/formats/rdrr/CATSCAN.md](../src/formats/rdrr/CATSCAN.md) | `doppler.runtime-source.formats` |
| `doppler.runtime-source.gpu.kernels` | Implement registered deterministic tensor operations whose numeric behavior and dispatch identity can be verified. | [src/gpu/kernels/CATSCAN.md](../src/gpu/kernels/CATSCAN.md) | `doppler.runtime-source.gpu` |
| `doppler.runtime-source.inference.pipelines` | Bind declared model types to pipeline implementations without changing normalized command or session semantics. | [src/inference/pipelines/CATSCAN.md](../src/inference/pipelines/CATSCAN.md) | `doppler.runtime-source.inference` |
| `doppler.runtime-source.inference.pipelines.text` | Execute manifest-declared transformer generation, embedding, reranking, and declared multimodal decoder workloads. | [src/inference/pipelines/text/CATSCAN.md](../src/inference/pipelines/text/CATSCAN.md) | `doppler.runtime-source.inference.pipelines` |

Child charters narrow inherited authority. They do not replace or broaden their parent contracts.
