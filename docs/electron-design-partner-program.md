# Electron design partner program

This program operationalizes Doppler's first external customer wedge without
converting research targets into product evidence.

The entry product is **Doppler Production Release**. The recurring product is
**Doppler Release Operations**. The initial buyer is a TypeScript/Electron
desktop team shipping a consequential local-model release across a declared
Windows and macOS fleet.

The validated target and pilot source is
[`electron-design-partner-prospects.json`](../tools/policies/electron-design-partner-prospects.json).
Its entries are prospects only. They do not belong in the product-integration
qualification registry until the application owner authorizes a pinned
application revision, workload, oracle, model release, fleet policy, and
evidence custody boundary.

## Outreach order

1. AnythingLLM: generation, RAG, retrieval, and provider compatibility.
2. Joplin: embedding, semantic search, index migration, and rollback.
3. Cherry Studio: generation, embedding, and model-provider capability.
4. Chatbox: backup knowledge-base, embedding, reranking, and provider release.
5. AFFiNE: later document-intelligence target, conditional on an
   owner-confirmed consequential local-model release.

Popularity research is a selection input, not a mutable field in Doppler's
qualification policy. Stars, downloads, release activity, and contact details
must be reverified before external use and never count as adoption.

## Pilot admission

A prospect may move from research to an authorized pilot only when Doppler has:

- explicit outreach and workload authorization;
- a pinned application revision and planned model release;
- application-owned acceptance tests and an independent oracle;
- a supported-device policy naming Electron, operating system, architecture,
  GPU, driver, and provider constraints;
- declared fallback, rollout, rollback, revocation, and support boundaries;
- written evidence-custody authority; and
- commercial terms that distinguish the initial release from paid recurring
  upgrade responsibility.

The prospect registry records relationship state. It never records a
qualification claim. Authorized execution evidence belongs in the existing
production-release, fleet-receipt, and product-integration contracts.

## Delivery sequence

```text
owner-authorized release input
  -> npx doppler release
  -> immutable candidate Pack
  -> application acceptance and provider controls
  -> customer-operated fleet qualification
  -> signed eligible-or-blocked recommendation
  -> customer activation or rejection
  -> monitoring, rollback, or revocation
  -> subsequent upgrade through Doppler Release Operations
```

Doppler never activates a release. The customer owns activation, rollback, and
deployment authority. A failed candidate is retained as evidence rather than
silently repaired or omitted.

## Provider and Doe boundary

The customer keeps its current runtime unless another provider wins the frozen
application acceptance contract. ONNX Runtime, WebLLM, Ollama, LM Studio,
Foundry Local, browser APIs, vendor accelerators, CPU references, or Doppler's
runtime may remain the executor.

DoeProof is an optional, separately authorized physical qualification provider.
It is not a mandatory stage and Doppler must remain sellable when Doe is absent
or loses. DoeRuntime is eligible only after the identical application test
shows a material win over the strongest incumbent. If an incumbent executes,
the Pack and release decision bind that exact provider identity and grant no
execution credit to Doe or Doppler.

Raw customer content never moves to Doe or another portfolio product by
default. Customer-derived evidence requires explicit authorization. Shared
learning is limited to the narrower sanitized or independently reproducible
classes declared by the prospect policy and the receiving product's custody
contract.

## Commercial exit

The first pilot establishes commercial evidence only when a customer pays for
or explicitly authorizes a consequential production release and uses Doppler's
signed recommendation in its promotion decision. Recurring evidence requires
the same customer to delegate a subsequent model upgrade to Doppler. Three
unrelated design partners require three independently authorized customer
relationships; the five-name research pipeline does not satisfy that gate.

Inspect the current pipeline with:

```bash
npm run product:prospects:report
```

Validate it with:

```bash
npm run product:prospects:check
```
