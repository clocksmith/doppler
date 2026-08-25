# Doppler Production Release platform

Doppler is an AI-native model release foundry for JavaScript and WebGPU. Its
initial product contract is deliberately narrow:

- Entry product: **Doppler Production Release**.
- Recurring product: **Doppler Release Operations**.
- Initial ICP: **TypeScript/Electron desktop products on Windows and macOS**.
- North star: **Production model releases whose eligibility decision is
  delegated to Doppler and relied upon by the customer’s activation system.**

Doppler signs an `eligible` or `blocked` recommendation. The customer’s updater,
deployment system, feature flag, or administrator retains authority to activate
and roll back the release. Doppler tooling and its GitHub Action must never
self-promote or deploy the customer application.

The validated contract is
`tools/policies/model-release-platform.json`. `npm run model-release:check`
binds that policy to the goal matrix and verifies the Forge/Pack/Runtime split,
the seven Pack release requirements, provider neutrality, Pack-first migration,
recovery, commercial boundary, and ordered promotion gates. A passing check
means the contract describes its gaps honestly; it does not establish external
adoption or production authority.

The repository implementation is closed for the canonical product contract,
reference release manifest, Pack release contract, Pack-first Electron
reranking, `doppler release`, the non-promoting GitHub Action, and the typed
Electron adapter. The policy separately reports gates that still require
customer-operated device receipts, a production revocation authority, and paid
external release evidence. Repository fixtures never satisfy those gates.

## Entry and recurring products

For one Doppler Production Release, a customer supplies:

- a pinned model revision, source and licence facts;
- a pinned Electron application revision and application-owned acceptance tests;
- an explicit workload and oracle identity;
- a supported Windows/macOS device policy, including Electron, GPU, and driver
  constraints;
- the incumbent provider control and data-custody rules;
- the previous working release, rollout rules, and rollback target.

Doppler returns an immutable Pack, a signed eligible-or-blocked release decision,
qualification receipts, typed exclusions, retained failure evidence, the
previous working Pack and rollback target, and revocation configuration.
Doppler Release Operations repeats that process for upgrades, requalification,
incidents, revocation, and support-fleet changes.

The immediate reference episode is an Electron document-search application
upgrading to `qwen-3-reranker-0-6b-q4k-ehf16-af32`. It is a checked-in acceptance
fixture, not evidence of external demand.

## Internal evidence and external proof

Reploid generation, Dream embedding retrieval, and Columbo reranking are:

> Internally controlled reference integrations proving application-integration
> mechanics.

Their 3/3 qualification establishes neither external adoption nor commercial
demand. Commercial proof requires one paid external Electron release whose
customer activation system relies on Doppler’s eligibility decision, followed
by a subsequent upgrade through the same process. Two additional unrelated
customers must then repeat the path.

The separate [Electron design partner program](electron-design-partner-program.md)
orders AnythingLLM, Joplin, Cherry Studio, Chatbox, and AFFiNE as researched
prospects. The validated register deliberately reports zero qualified customers.
It cannot satisfy this section's external, paid, fleet, rollback, revocation, or
repeat-upgrade gates.

## Pack release closure

The immutable Pack is the supported release unit. Pack v2 now binds and rejects
drift across all seven release elements:

1. Source revision, licence, and provenance.
2. Application workload and oracle identities.
3. Known exclusions and typed rejections.
4. Version supersession and migration.
5. Revocation policy.
6. Failed-upgrade preservation.
7. Portable state-snapshot identity.

Complete immutable release binding is the enclosing requirement, not an eighth
element. This closes the repository Pack representation; it does not qualify a
production revocation authority or supply external customer evidence.

## Pack-first production path

The first production path is Electron reranking. It must validate the Pack,
select a qualified TargetPlan for the exact device tuple, bind a SessionPlan,
execute the application workload, and retain application and fleet evidence.
The runtime now exposes this narrow path as a Pack-bound rerank session: the
caller must present the exact application revision, workload, and oracle
identities signed into the Pack, and the result is a receipt bound to the Pack,
selected TargetPlan, lifecycle, and revocation policy. Repository tests use a
mock program and therefore establish contract behavior, not WebGPU fleet
qualification.

Application-gate evidence must identify the Pack semantic root, selected
TargetPlan, resolved execution digest, provider, and exact device target in
addition to the application revision, workload, oracle, evaluator, quality,
latency, memory, startup, recovery, and failed samples. The qualification agent
passes the Pack, target, and device paths through explicit environment fields
and rejects a receipt that reports a different execution. A generic application
smoke receipt cannot be wrapped into fleet evidence.
Dynamic model loading remains an explicit intake/conversion compatibility
surface; it cannot bypass production qualification. Generic OpenAI, generation,
browser expansion, and unrelated embedding surfaces are not on the immediate
gate unless the external Electron release requires them.

Chromium/WebGPU correctness remains required because Electron executes WebGPU
in its renderer. A hosted GitHub runner may orchestrate customer-operated
Windows and macOS qualification agents, but cannot stand in for the supported
fleet.

Each `supportedDevices.targets` entry is one qualification tuple. It declares
one OS, one architecture, one GPU vendor, an exact GPU-device allowlist, an
exact driver-version allowlist, an Electron version range, and one qualification
surface. The device receipt also carries the observed `shader-f16`, subgroup,
and maximum-buffer capabilities used to select an already-qualified TargetPlan.
A signed receipt must match every declared dimension; a receipt for one tuple
never covers a second tuple by sharing a vendor. Target IDs and rollout stages
are unique, rollout percentages advance monotonically to 100%, and the rollback
target must be the pinned previous release.

The public command forms are `doppler release` and
`npx --package doppler-gpu doppler release`; the npm package is named
`doppler-gpu`. The command has explicit `qualify` and `decide` phases. It emits
signed fleet receipts or an eligible/blocked decision plus immutable evidence,
including a retained copy of the exact candidate Pack envelope, and always
reports `activationPerformed: false`. The reusable workflow preserves every
downloaded artifact but passes only schema-identified Electron fleet receipts
to the decision command; application receipts, device inputs, and failure JSON
cannot be mistaken for fleet qualification. It fails its check when eligibility
is blocked.

## Provider and custody boundaries

Runtime selects only prequalified TargetPlans and cannot invent a plan or
fallback on the customer device. Doe and incumbent providers remain eligible
only when explicitly qualified for the application contract.

Customer-private models, inputs, outputs, application facts, activation state,
and rollback authority remain customer-controlled unless explicit authority
grants a narrower use. Shared learning is limited to sanitized failure
signatures, minimized synthetic reproductions, public evidence, and
customer-approved derived patterns.

External release dependence is the operating objective. Acquisition interest
is a possible consequence, not a substitute for completing the release path.
