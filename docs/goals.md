# Doppler Goals

This file is the compact product and technical contract for Doppler's mainline
work. Generated matrices and receipts remain the source of truth for current
support and claims; this file states what the repo is optimizing toward.

## Goal 1: Make local WebGPU inference a real product surface

Product contract:

- Users can run verified local inference through the hosted browser demo,
  `npx doppler-gpu`, the root `doppler` API, CLI surfaces, Node/Bun execution,
  and the OpenAI-compatible localhost server.
- Tier 1 behavior is the verified text-inference path behind those surfaces.
- Advanced exports may exist, but support tier is defined by
  `docs/subsystem-support-matrix.md`, not by export shape alone.

Technical contract:

- One JS/WGSL WebGPU inference engine serves browser, Node, Bun, CLI, and server
  surfaces.
- Command semantics stay aligned across browser and CLI runners.
- JavaScript orchestrates load, prefill, decode, KV cache, streaming, and
  readback from resolved config.
- WGSL kernels perform deterministic math only.
- Unsupported environment or runtime capabilities fail closed instead of
  silently falling back.

## Goal 2: Own the model artifact and runtime contract

Product contract:

- Users can load verified hosted models by registry ID or explicit model URL.
- The model support matrix states which model families, surfaces, and receipts
  are verified.
- Local caches are implementation details; artifact identity is still manifest
  and catalog owned.

Technical contract:

- RDRR manifests own model parameters, shard identity, tokenizer metadata,
  quantization, session policy, execution graph, kernel references, dtype policy,
  and artifact identity.
- Conversion-owned storage facts require conversion or manifest migration to
  change. Runtime config may not rewrite them.
- Runtime config may overlay only runtime-owned execution policy such as session,
  loading, diagnostics, and explicit kernel-path/session behavior.
- Defaults must be represented in schema, manifest, profile, config, or rule
  assets. Runtime code must not invent hidden behavior.

## Goal 3: Make correctness and performance evidence-backed

Product contract:

- Doppler says "this model works" only from verified model/support evidence.
- Doppler says "this lane is faster/slower" only from comparable benchmark
  artifacts with clean correctness and explicit disclosure.
- Candidate, diagnostic, experimental, and internal-only lanes remain visible but
  are not promoted as tier 1 product claims.

Technical contract:

- Benchmarks use shared workload contracts, command parity, normalized compare
  JSON, release matrices, and traceable artifact paths.
- Kernel paths, dtype policy, batching, readback cadence, and runtime profiles
  are explicit and receipt-visible.
- Claimable comparisons require prompt/sampling/cache/load alignment, non-zero
  decode, clean correctness, real artifact source identity, and one traceable
  artifact per claim.
- Debug and investigation profiles are useful evidence, but not public speed
  claims unless promoted by the benchmark policy.

## Ownership map

| Area | Goal |
| --- | --- |
| Browser demo, CLI, server, root API, Node/Bun surfaces | Goal 1 |
| JS orchestration, WGSL kernels, KV cache, streaming | Goal 1 |
| RDRR, conversion, manifests, catalog, hosted model IDs | Goal 2 |
| Runtime profiles, schema defaults, rule maps, kernel refs | Goal 2 |
| Release receipts, model matrix, subsystem tiers | Goal 3 |
| Vendor compares, benchmark SVGs, claim matrices | Goal 3 |

## Non-goal boundary

Doppler is the inference engine and artifact/runtime contract. Orchestrators,
agents, product loops, and policy layers decide what to ask, when to ask it, and
how to use the result. Doppler executes explicit contracts; it does not infer
policy from application intent.
