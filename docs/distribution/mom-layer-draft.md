# MoM Layer Draft

Doppler implementation-side draft for supporting the Mixture of MoEs (MoM)
proposal.

The proposal defines the narrative and architecture. This document defines what
Doppler would need to ship under that proposal: cross-family routing, substrate
execution receipts, debate/adjudication records, recursion receipts, residency
metadata, and distillation export hooks.

## Status

- spec status: implementation-side draft
- implementation status: not shipped as a complete runtime surface
- current substrate pieces:
  - [`transcriptRoot`](transcript-root.md)
  - [`peer identity`](peer-identity-contract.md)
  - [`p2p control plane`](p2p-control-plane-extensions.md)
  - [`RDRR P2P execution plan`](rdrr-p2p-plan.md)
  - `doppler-gpu/orchestration` experimental exports for routers, pipeline pools,
    KV cache helpers, and logit merge helpers

## Non-goals

- Do not make this document the canonical MoM proposal.
- Do not redefine internal MoE routing inside one model.
- Do not replace distributed inference plans.
- Do not mutate `artifactIdentity` with request-local state.
- Do not perform runtime plan synthesis for distributed tensor placement.
- Do not silently fall back to a different substrate when the selected substrate
  cannot produce the required receipt.

## Terms

- MoM layer: Doppler's implementation surface for the MoM proposal.
- Cross Family Router (CFR): the router that selects substrate families and
  execution roles for a request.
- Debate Weighted Distillation (DWD): the feedback path that turns adjudicated
  debate records into distillation candidates.
- Proposal, Trace, Adjudication (PTA): the minimum receipt shape for one MoM
  decision.
- substrate: a runnable model family, adapter stack, pipeline pool, or nested MoM.
- substrate transcript root: the lower-layer `transcriptRoot` for one substrate
  execution.
- MoM transcript root: the root binding CFR input, substrate transcript roots,
  debate records, adjudication, and distillation export metadata.
- receipt budget: the bounded resource that controls recursive MoM expansion.

## Relationship to distributed inference

Distributed inference answers:

```text
How does one canonical artifact run across peers while preserving parity?
```

Doppler's MoM layer answers:

```text
Which substrate family should handle this request, which peers produced which
proposals, which proposal won, and what receipt lets the route be replayed or
distilled?
```

The layering is:

```text
request
  -> CFR route plan
  -> substrate executions
       -> distributed inference plans
       -> substrate transcript roots
  -> debate/adjudication
  -> MoM transcript root
  -> DWD export candidate
```

The lower layer can prove that one selected substrate executed as declared. The
MoM-layer receipts prove why that substrate was selected, what alternatives
participated, what each one produced, how the winner was chosen, and what can be
distilled.

## MoM route plan

The route plan is authored before substrate execution. It must be replayable from
declared inputs and checked policy. It must not depend on hidden runtime
branching.

Shape:

```json
{
  "momVersion": 0,
  "requestId": "req_...",
  "inputHash": "sha256:...",
  "receiptBudget": {
    "units": 8,
    "spent": 0,
    "policyHash": "sha256:..."
  },
  "cfr": {
    "policyId": "cfr.default.v0",
    "policyHash": "sha256:...",
    "tiePolicy": "stable-choice",
    "candidateSetHash": "sha256:...",
    "selected": [
      {
        "substrateId": "gemma4-moe-local",
        "role": "proposer",
        "weight": 0.58,
        "reasonCodes": ["resident", "tool-fit", "cost-fit"]
      }
    ]
  },
  "substrates": [
    {
      "substrateId": "gemma4-moe-local",
      "family": "gemma4",
      "artifactIdentityHash": "sha256:...",
      "manifestHash": "sha256:...",
      "adapterHash": "sha256:...",
      "prefixDigest": "sha256:...",
      "residency": "hot",
      "capabilitiesHash": "sha256:...",
      "distributedPlanId": "lan-4peer-tp"
    }
  ],
  "debate": {
    "rounds": [],
    "adjudicator": {
      "substrateId": "judge-local",
      "policyHash": "sha256:..."
    }
  },
  "recursion": {
    "parentMomTranscriptRoot": null,
    "childMomTranscriptRoots": []
  }
}
```

Required invariants:

- `momVersion` is explicit.
- `inputHash` binds the request payload after canonicalization.
- `receiptBudget.policyHash` binds the recursion budget policy.
- `cfr.policyHash` binds the exact routing policy.
- `cfr.tiePolicy` is explicit.
- every selected substrate has an `artifactIdentityHash`.
- every selected substrate declares residency.
- every selected substrate either declares a distributed plan or explicitly runs
  solo.
- missing selected-substrate receipts fail closed.

## JavaScript route-plan sketch

This is the proposed host-orchestrator shape, not a shipped `doppler-gpu` API.
The important part is the contract: routing is data-driven, every route input is
hashed, and missing receipts fail before execution continues.

```js
const RESIDENCY_RANK = {
  hot: 3,
  warm: 2,
  cold: 1,
  remote: 0,
};

function stableTieKey(candidate) {
  return [
    candidate.substrateId,
    candidate.artifactIdentityHash,
    candidate.manifestHash,
    candidate.adapterHash ?? '',
  ].join('\0');
}

function requireDigest(value, field) {
  if (typeof value !== 'string' || !value.startsWith('sha256:')) {
    throw new Error(`MoM CFR candidate missing ${field}`);
  }
}

function scoreCandidate(candidate, policy) {
  requireDigest(candidate.artifactIdentityHash, 'artifactIdentityHash');
  requireDigest(candidate.manifestHash, 'manifestHash');
  requireDigest(candidate.capabilitiesHash, 'capabilitiesHash');

  const residency = RESIDENCY_RANK[candidate.residency];
  if (residency == null) {
    throw new Error(`Unsupported MoM residency class: ${candidate.residency}`);
  }

  const weights = policy.weights;
  const toolFit = candidate.toolFamilies?.includes(policy.requiredToolFamily) ? 1 : 0;
  const contextFit = Math.min(candidate.maxContextTokens / policy.requiredContextTokens, 1);

  return (
    residency * weights.residency +
    toolFit * weights.toolFit +
    contextFit * weights.contextFit -
    candidate.estimatedCost * weights.cost
  );
}

export function buildMomRoutePlan({ request, candidates, policy, receiptBudget }) {
  if (!Array.isArray(candidates) || candidates.length === 0) {
    throw new Error('MoM CFR requires at least one candidate substrate');
  }
  if (receiptBudget.units <= receiptBudget.spent) {
    throw new Error('MoM receipt budget exhausted before routing');
  }

  const ranked = candidates
    .map((candidate) => ({
      candidate,
      score: scoreCandidate(candidate, policy),
      tieKey: stableTieKey(candidate),
    }))
    .sort((a, b) => b.score - a.score || a.tieKey.localeCompare(b.tieKey));

  const selected = ranked.slice(0, policy.topK).map(({ candidate, score }, index) => ({
    substrateId: candidate.substrateId,
    role: index === 0 ? 'proposer' : 'critic',
    weight: score,
    reasonCodes: [
      candidate.residency,
      candidate.toolFamilies?.includes(policy.requiredToolFamily) ? 'tool-fit' : 'tool-miss',
    ],
  }));

  return {
    momVersion: 0,
    requestId: request.requestId,
    inputHash: request.inputHash,
    receiptBudget,
    cfr: {
      policyId: policy.policyId,
      policyHash: policy.policyHash,
      tiePolicy: 'stable-choice',
      candidateSetHash: request.candidateSetHash,
      selected,
    },
    substrates: selected.map((entry) => {
      const candidate = candidates.find((item) => item.substrateId === entry.substrateId);
      return {
        substrateId: candidate.substrateId,
        family: candidate.family,
        artifactIdentityHash: candidate.artifactIdentityHash,
        manifestHash: candidate.manifestHash,
        adapterHash: candidate.adapterHash ?? null,
        prefixDigest: candidate.prefixDigest ?? null,
        residency: candidate.residency,
        capabilitiesHash: candidate.capabilitiesHash,
        distributedPlanId: candidate.distributedPlanId ?? null,
      };
    }),
    debate: {
      rounds: [],
      adjudicator: {
        substrateId: policy.adjudicatorSubstrateId,
        policyHash: policy.adjudicationPolicyHash,
      },
    },
    recursion: {
      parentMomTranscriptRoot: request.parentMomTranscriptRoot ?? null,
      childMomTranscriptRoots: [],
    },
  };
}
```

## JavaScript debate sketch

This sketch shows how proposal hashes, substrate transcript roots, and
adjudication get bound into one MoM transcript. A real implementation must use
Doppler's canonical JSON hasher instead of carrying its own local copy.

```js
const textEncoder = new TextEncoder();

function canonicalJson(value) {
  if (value == null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(',')}]`;
  }
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(',')}}`;
}

async function sha256Json(value) {
  const bytes = textEncoder.encode(canonicalJson(value));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const hex = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, '0')
  ).join('');
  return `sha256:${hex}`;
}

export async function runMomDebate({ routePlan, executeSubstrate, adjudicate }) {
  const proposals = [];

  for (const selected of routePlan.cfr.selected) {
    const result = await executeSubstrate(selected.substrateId, routePlan);
    if (!result.substrateTranscriptRoot) {
      throw new Error(`Missing substrate transcript root for ${selected.substrateId}`);
    }

    proposals.push({
      substrateId: selected.substrateId,
      payloadType: result.payloadType,
      proposalHash: await sha256Json(result.payload),
      substrateTranscriptRoot: result.substrateTranscriptRoot,
      confidence: result.confidence,
    });
  }

  const round = {
    round: 0,
    participants: routePlan.cfr.selected.map((entry) => entry.substrateId),
    proposals,
    critiques: [],
  };

  const adjudication = await adjudicate({ routePlan, round });
  const transcript = {
    momVersion: routePlan.momVersion,
    routePlanHash: await sha256Json(routePlan),
    cfrPolicyHash: routePlan.cfr.policyHash,
    substrateTranscriptRoots: proposals.map((proposal) => proposal.substrateTranscriptRoot),
    debateRoundHashes: [await sha256Json(round)],
    adjudicationHash: await sha256Json(adjudication),
    recursion: routePlan.recursion,
    dwdExportHash: null,
  };

  return {
    round,
    adjudication,
    momTranscriptRoot: await sha256Json(transcript),
  };
}
```

## Cross Family Router

CFR chooses substrate families, not individual MoE experts inside one model. It
consumes:

- request features
- model and adapter capabilities
- declared residency
- policy constraints
- device and peer capability summaries
- previous route receipts when available

It emits:

- candidate set hash
- selected substrates
- role assignment
- route weights
- reason codes
- tie policy
- policy hash

CFR must not:

- keyword-route in code
- rewrite substrate identity at runtime
- select an undeclared fallback substrate
- use load or residency as an unreceipted hidden input

## Residency contract

The MoM layer needs manifest-based residency so hot and cold clusters are
explicit.

Residency classes:

- `hot`: resident weights or adapters are already loaded and validated.
- `warm`: artifact is locally addressable but needs activation, adapter swap, or
  KV/prefix materialization.
- `cold`: artifact is known but must be fetched or mounted before use.
- `remote`: artifact runs on a declared peer or provider boundary.

CFR may prefer hot or warm substrates, but the route receipt must record the
class used for selection. If a substrate changes class during a run, the run must
record a new capability summary hash before CFR consumes it.

## Debate protocol

Debate is an explicit exchange over substrate outputs, not an implicit output
merge.

Each debate round records:

```json
{
  "round": 0,
  "participants": ["gemma4-moe-local", "qwen-moe-remote"],
  "proposals": [
    {
      "substrateId": "gemma4-moe-local",
      "proposalHash": "sha256:...",
      "substrateTranscriptRoot": "sha256:...",
      "confidence": 0.74
    }
  ],
  "critiques": [
    {
      "fromSubstrateId": "qwen-moe-remote",
      "targetProposalHash": "sha256:...",
      "critiqueHash": "sha256:..."
    }
  ]
}
```

The debate record must bind proposal bytes by hash. The proposal can be text,
structured JSON, tool arguments, logits summary, or another typed payload, but
the payload type must be declared by the caller-facing command contract.

## Adjudication

Adjudication records the winner and the reason the winner is accepted.

Shape:

```json
{
  "winnerProposalHash": "sha256:...",
  "winnerSubstrateId": "gemma4-moe-local",
  "adjudicatorSubstrateId": "judge-local",
  "policyHash": "sha256:...",
  "scoreVectorHash": "sha256:...",
  "rejects": [
    {
      "proposalHash": "sha256:...",
      "reasonCodes": ["policy-miss", "low-evidence"]
    }
  ]
}
```

Adjudication must be deterministic under fixed inputs, policy, and declared
tie policy. If the adjudicator is itself a model, its execution needs a substrate
transcript root.

## MoM transcript root

The MoM transcript root is the replay root for the whole orchestration. It is not
the same as a lower-layer substrate `transcriptRoot`.

Minimum fields:

- route plan hash
- CFR policy hash
- substrate transcript roots
- debate round hashes
- adjudication hash
- recursion parent and child roots
- residency summary hash
- DWD export hash or explicit `null`

Replay claim:

- `MoM transcript root + request payload + declared substrate artifacts -> exact
  orchestration replay`

The replay claim is invalid if any selected substrate lacks a transcript root.

## Recursion contract

Doppler's MoM-layer recursion is controlled by receipt budget, not a hidden
depth cap.

Rules:

- A nested MoM consumes budget before it can launch.
- A nested MoM has its own MoM transcript root.
- The parent records each child root.
- The child records its parent root.
- Exhausted budget fails closed before launching the child route.

The budget unit is policy-defined. A policy may charge per selected substrate,
per debate round, per remote peer, per tool boundary, or per nested MoM.

## Debate Weighted Distillation

DWD exports adjudicated records as training candidates. It does not update a
model during the request path.

Minimum export row:

```json
{
  "dwdVersion": 0,
  "momTranscriptRoot": "sha256:...",
  "winnerProposalHash": "sha256:...",
  "winnerSubstrateId": "gemma4-moe-local",
  "teacherSetHash": "sha256:...",
  "studentTarget": {
    "family": "gemma4",
    "adapterId": "route-adapter-v0"
  },
  "scoreVectorHash": "sha256:...",
  "accepted": true
}
```

DWD rows are candidates until the training pipeline validates them, records
lineage, and promotes the resulting artifact through the normal model gates.

## Promotion gates

Doppler's MoM layer cannot be called shipped until these gates exist:

1. route-plan schema validation
2. CFR deterministic replay fixtures
3. residency summary hash validation
4. debate record validation
5. adjudication replay validation
6. MoM transcript root verifier
7. nested MoM receipt-budget rejection fixture
8. DWD export validation
9. substrate transcript-root coverage check
10. command parity across browser and Node surfaces

## Implementation notes

- `MoERouter` remains the internal router for one MoE model.
- CFR lives above `MoERouter`.
- `MultiModelNetwork` and `MultiPipelinePool` are useful substrate execution
  primitives but do not define the MoM contract.
- distributed `plans[]` and `transcriptRoot` remain lower-layer contracts.
- signed peer envelopes belong to the peer identity layer; MoM records their
  roots and participant identities.

## Open questions

- exact canonical JSON shape for `momVersion: 0`
- whether CFR policy lives in RDRR metadata, runtime config, or a separate route
  policy artifact
- how much logit data a proposal receipt should carry
- whether DWD exports live under the training artifact policy or a new replay
  corpus namespace
- whether remote provider substrates can satisfy the same transcript-root
  standard or need a weaker declared receipt class
