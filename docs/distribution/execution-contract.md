# Distributed RDRR execution contract

Additive distributed execution contract for canonical RDRR artifacts.

This is a design contract, not an implementation queue.

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
   - reproduces one distributed execution of that artifact

Replay claim:
- `artifactIdentity + transcriptRoot + input tokens -> exact distributed replay`

Without `transcriptRoot`, only the declared determinism class is reproducible.

### `transcriptRoot` fields

Minimum contract surface:
- token schedule
- RNG state placement
- per-stage output hashes
- peer assignments
- activation Merkle roots
- KV Merkle roots

The coordinator authors this object; replay verification consumes it.

MoM orchestration composes multiple substrate transcript roots into a separate
MoM transcript root. This contract defines only the lower-layer substrate
execution receipt. See [MoM Layer Draft](mom-layer-draft.md) for the
cross-family route, debate, adjudication, recursion, and distillation receipt.

## Contract requirements

### Integrity and addressability

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

### Exact distributed execution

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

### Topology-adaptive selection

One artifact, multiple named plans.

Runtime responsibilities:
- peers publish capability + budget summaries
- coordinator selects one precomputed compatible plan

Non-goal:
- no runtime plan synthesis
- no runtime weight rewriting

Promotion gate:
- every shipped plan passes parity against the canonical exact plan

### Expert identity

MoE distribution changes expert placement, not expert choice.

Rules:
- router output must match solo exactly
- `dispatchPlan.routingPolicy = "exact"` is required in the parity lane
- locality-biased or capacity-biased routing is not part of Phase 3 parity

Promotion gate:
- MoE distributed vs solo parity

### Recovery

See [Recovery semantics](#recovery-semantics).

### Performance and exactness

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

## Peer identity

Every cross-peer tensor envelope must be attributable to one session peer
without creating durable machine identity.

- key type: `ed25519`
- scope: one distributed session
- durability: non-durable
- signer metadata includes peer id + session id

Envelope verification must reject a forged signer, a wrong session, and an
expired session.

## Control-plane session

The control plane is an additive protocol for distributed plan selection and
session execution.

Session states: `join`, `ready`, `run`, `complete`, `abort`.

Responsibilities:
- coordinator assignment protocol
- peer introspection payload collection
- distributed plan selection over `plans[]`
- ingress routing to stage-0 owner
- egress routing from final-stage owner

Shard-delivery control-plane hooks (session tokens, policy evaluation) are in
the [P2P transport contract](p2p-transport-contract.md#control-plane-boundary).

## Recovery semantics

Peer churn and Byzantine behavior are handled explicitly, and recovery depends
on declared redundancy:

- replicated stage (`N >= 2`) or replicated collective path: isolate the bad
  peer, then retry or redirect to a replica under plan policy, and continue
- non-replicated stage or path: fail closed with
  `DOPPLER_DISTRIBUTED_STAGE_UNRECOVERABLE`
- no implicit survivability claims; neither an integrity mismatch nor the
  collective transport contract implies survivability on its own

Promotion gate:
- induced peer drop / Byzantine vector tests complete correctly or fail closed

## Related contracts

- [collective-transport-contract.md](collective-transport-contract.md)
- [p2p-transport-contract.md](p2p-transport-contract.md)
- [../rdrr-format.md](../rdrr-format.md)
