# RDRR P2P Execution Plan

Additive distributed execution contract for canonical RDRR artifacts.

## Purpose

Keep one canonical RDRR weight artifact while allowing Doppler to execute that
artifact:
- solo
- on a LAN cluster
- across a WAN swarm

Distribution is an additive capability, not a parallel weight format.

## Non-goals

- no topology-specific repacked weight files
- no runtime placement synthesis
- no approximation in the parity lane
- no mutation of `artifactIdentity` to include request-local state

## Two roots, not one

Distributed execution uses two distinct identities:

1. `artifactIdentity`
   - immutable
   - binds canonical weights, manifest, execution graph, and additive integrity extensions
   - reproduces the model artifact

2. `transcriptRoot`
   - per request
   - binds token schedule, RNG placement, activation/KV roots, peer assignments, and stage outputs
   - reproduces one distributed execution of that artifact

Replay claim:
- `artifactIdentity + transcriptRoot + input tokens -> exact distributed replay`

Without `transcriptRoot`, only the declared determinism class is reproducible.

## Phase structure

### Phase 0: integrity + addressability

Additive to canonical RDRR:
- `manifest.integrityExtensions`
- tensor physical storage descriptors
- deterministic slice compilation against canonical shard bytes

Current implementation boundary:
- `integrityExtensions.contractVersion = 1`
- `integrityExtensions.blockMerkle.blockSize`
- `integrityExtensions.blockMerkle.roots`

Promotion gate:
- solo load must remain byte-identical to pre-Phase-0 artifacts

### Phase 1: exact distributed execution

Ship `distributed.json` with one exact plan.

Target:
- tensor parallel within LAN
- pipeline parallel across WAN

Correctness claim:
- correctness-shippable only
- WAN latency-shippable is explicitly deferred to Phase 5 overlap/microbatching

Promotion gate:
- distributed logits vs solo logits
- every collective step must be `algorithm-exact` or stricter

### Phase 2: topology-adaptive plan selection

One artifact, multiple named plans.

Runtime responsibilities:
- peers publish capability + budget summaries
- coordinator selects one precomputed compatible plan

Non-goal:
- no runtime plan synthesis
- no runtime weight rewriting

Promotion gate:
- every shipped plan passes parity against the canonical exact plan

### Phase 3: exact MoE expert distribution

MoE distribution changes expert placement, not expert choice.

Rules:
- router output must match solo exactly
- `dispatchPlan.routingPolicy = "exact"` is required in the parity lane
- locality-biased or capacity-biased routing is not part of Phase 3 parity

Promotion gate:
- MoE distributed vs solo parity

### Phase 4: robustness

Peer churn and Byzantine behavior are handled explicitly.

Rules:
- replicated stage: isolate bad peer and continue
- non-replicated stage: fail closed
- no implicit survivability claims

Promotion gate:
- induced peer drop / Byzantine vector tests complete correctly or fail closed

### Phase 5: speed lanes

Optimization-only phase:
- activation compression
- collective/compute overlap
- microbatch shaping
- locality-biased expert routing

All Phase 5 approximations must declare a tolerance-bounded exactness class and
are rejected in the parity lane.

## `distributed.json` contract

The distributed execution plan is a sibling artifact, not core `manifest.json`.

Example shape:

```json
{
  "rdrd": 1,
  "compatibility": {
    "artifactIdentityHash": "sha256:...",
    "manifestHash": "sha256:...",
    "executionGraphDigest": "sha256:...",
    "integrityExtensionsHash": null,
    "synthesizerVersion": "doppler-placement@1.4.0",
    "synthesizedAt": "2026-04-23T00:00:00.000Z"
  },
  "plans": [
    {
      "id": "lan-4peer-tp",
      "topologyHash": "sha256:...",
      "topology": {},
      "stages": [],
      "tensorPlacements": [],
      "collectives": [],
      "prefetch": []
    }
  ]
}
```

Validation rule:
- loader validates `compatibility` before any collective is established
- mismatch fails closed with `DOPPLER_DISTRIBUTED_PLAN_STALE`

## Compatibility hashing contract

Every hash in `distributed.json` uses canonical JSON:
- UTF-8 encoding
- recursively lexicographically sorted object keys
- arrays preserved in declared order
- no insignificant whitespace
- `null` kept explicit
- `undefined` omitted
- finite JSON numbers only

Hash meanings:
- `artifactIdentityHash`: canonical hash of `manifest.artifactIdentity`
- `manifestHash`: canonical hash of `manifest.json`
- `executionGraphDigest`: canonical hash of `manifest.inference.execution`
- `integrityExtensionsHash`:
  - `null` when the target artifact has no `integrityExtensions`
  - exact canonical hash of `manifest.integrityExtensions` when present
- `topologyHash`: canonical hash of the normalized topology input used by the synthesizer

## Collective exactness classes

Each collective-bearing step in `executionGraphV2` declares one exactness class.

### `bit-exact-solo`

Bitwise identical to solo execution.

Use:
- conformance lanes only

### `algorithm-exact`

Bitwise identical when the declared reduction schedule is followed exactly.

Required metadata:
- reduction order
- tree/ring shape
- accumulation dtype

This is the Phase 1 default.

### `tolerance-bounded`

Declared approximation class with explicit metric and epsilon.

Use:
- Phase 5 only

Parity rule:
- parity lane rejects `tolerance-bounded`

## Prefetch ownership

Artifact-level hints stay minimal and solo-friendly.

Core RDRR may carry only intrinsic hints such as first-use metadata.

Plan-local prefetch is owned by `distributed.json.plans[].prefetch` because it
depends on placement and peer assignment.

Example:

```json
{
  "peer": "p0",
  "schedule": [
    {
      "step": "embed.lookup",
      "byteRanges": [],
      "priority": 0
    },
    {
      "step": "lm_head.proj",
      "byteRanges": [],
      "priority": 2,
      "reuse": true
    }
  ]
}
```

These byte ranges are pre-resolved offline by the synthesizer against canonical
shards. Runtime does not re-solve logical slices heuristically.

## Recovery semantics

Distributed recovery is explicit:

- replicated stage (`N >= 2`): isolate peer, redirect to replica, continue
- non-replicated stage: fail closed with `DOPPLER_DISTRIBUTED_STAGE_UNRECOVERABLE`

Integrity mismatch never implies implicit survivability.

## Related contracts

- [collective-transport-contract.md](collective-transport-contract.md)
- [p2p-transport-contract.md](p2p-transport-contract.md)
- [../rdrr-format.md](../rdrr-format.md)
