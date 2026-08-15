# Product Integration Qualification

Product integration qualification proves that a maintained application delivers
one declared Doppler workload as a supportable user outcome. Static application
code, a demo, a model load, or a successful isolated inference does not satisfy
this gate.

## Authority

`tools/policies/product-integration-qualification.json` is the status authority.
`npm run product:integrations:check` validates the policy, retained receipts,
receipt identities, and regression tests. The initial portfolio requires three
distinct active applications spanning generation, embedding retrieval, and
reranking.

Every evidence entry is either null or `{ path, digest }`. `path` is repository-
relative JSON; `digest` is the canonical JSON SHA-256. The checker recomputes
the digest, so changing a receipt without updating the governed reference fails
closed. Existing files, Markdown narratives, and discovery audits are not
qualification receipts.

## Ownership and execution identity

`ownerConfirmation` uses
`doppler.product-integration-owner-confirmation/v1`. It binds the application,
workload, named owner, owner repository, exact application revision,
maintenance state, and confirmation time. That time must equal the policy's
`ownerConfirmedAtUtc` and remain within the configured age limit.

`identity` is a native local `doppler_provider_receipt_v1`. Its logical model,
resolved manifest SHA-256, and resolved execution SHA-256 must exactly match the
integration. A fallback, failed, unresolved, or environment-free receipt is
retained evidence but cannot qualify the integration.

## Product outcome evidence

The remaining fields use `doppler.product-integration-evidence/v1`. Every
receipt binds the integration, owner, application and harness revisions,
environment fingerprint, workload, exact Doppler identities, capture time, and
class-specific observations. The checker derives the result from those
observations and rejects a claimed pass that contradicts them.

| Evidence class | Required acceptance evidence |
| --- | --- |
| `installToFirstVerifiedOutput` | Install succeeds and measured first verified output stays within its declared limit. |
| `sourceTaskQualityRetention` | Doppler/source score ratio is internally consistent and meets the frozen minimum. |
| `reliability` | Success rate and crash, OOM, and device-loss counts satisfy their declared limits. |
| `memory` | Measured peak bytes stay within the declared application budget. |
| `coldWarmResponse` | Retained sample count and cold/warm p50 and p95 values satisfy declared p95 limits. |
| `browserHardwareQualification` | The declared minimum target count qualifies with no failed declared target. |
| `incumbentControl` | A digest-bound incumbent comparison exists and performed correctness-comparable work. |
| `upgradeRequalification` | Migration, identity preservation, and the task gate all pass across named versions. |
| `rollbackRevocation` | Rollback to a known-safe version, revocation observation, and the post-rollback task gate pass. |

Receipts must not postdate `qualifiedAtUtc`. A structurally valid failed receipt
stays visible and contributes a named qualification blocker. It must never be
deleted or rewritten into a pass.

## Recording an evaluation

`npm run product:integrations:record -- --capture <capture.json> --out
<candidate-policy.json>` records one retained evaluation. The capture uses
`doppler.product-integration-evaluation-capture/v1` and supplies repository-
relative receipt paths, the evaluation time, and expiry. The recorder derives
canonical digests, the owner-confirmation time, and runtime-observed artifact
and execution identities from those receipts; operators do not retype them.

The default writes a separate candidate policy. Writing the status authority
requires explicit `--apply`, replacing prior evaluation state requires
`--replace`, and claimable entries cannot be replaced. Recording preserves the
declared qualification level, forces candidate lifecycle, retains semantic
failure reasons as blockers, and always leaves `claimAllowed` false. Promotion
is a separate review and policy change.

## Product gate

A claimable integration must be `product-supported`, active, owned by a current
maintainer, unexpired, identity-complete, free of blockers, and backed by all
eleven digest-bound receipts. Three endpoints or demos from one application do
not satisfy the distinct-application requirement.

The candidate audit in
`docs/status/product-integration-candidate-audit-2026-08-15.json` records only
static discovery. It deliberately supplies none of these receipts and leaves
all candidates non-claimable.
