# Executable Pack identity and release events

Pack v2 remains readable. Its timestamp and release fields remain in its
historical semantic root. Never rewrite or re-sign a historical v2 document in
place. Pack v3 contains only the executable ModelIR, TargetPlans, WGSL modules,
artifact inventory, program, model ID, root, and signature. Release promotion
does not change that executable root; re-signing changes the envelope digest.

## Migration and public API

The new source API is exported as `doppler-gpu/pack`. It is not a claim that a
published package, customer release, or peer network already consumes v3.

```js
import { migratePackV2, getPackIdentity, signPackReleaseEvent } from 'doppler-gpu/pack';

const migrated = await migratePackV2(oldPack, { trustedSigners, signer });
const { schema, semanticRoot, envelopeDigest } = getPackIdentity(migrated.pack);
const event = await signPackReleaseEvent({
  pack: { schema, semanticRoot, envelopeDigest },
  sequence: 1,
  previousEventDigest: null,
  issuedAtUtc,
  expiresAtUtc,
  action: 'eligible',
  release: migrated.release,
  migratedFrom: migrated.migratedFrom,
  nextSigner: null,
}, releaseSigner);
```

Migration verifies the old signature and signs the unchanged executable closure.
It does not manufacture source truth, byte verification, new qualification,
application acceptance, or production release authority. Eligibility is a
separately authorized act. The old envelope digest is retained as provenance.

## Opening and trust

`doppler-gpu` and `doppler-gpu/runtime` retain explicit device, artifact-store,
signer, and program-factory ports. `doppler-gpu/compat` exposes the application
`openPack` facade with the built-in program adapter. Both use the same verifier.
An artifact source must implement `readArtifact(descriptor)` returning bytes.
Runtime recomputes each SHA-256 and size and supplies copies from verified storage
to execution. A lying hash callback or manifest reference outside the signed
inventory cannot admit different bytes. Discovery and source assignment remain
Reploid's responsibility; this interface does not implement a swarm or seeder.

For v3, pass `releaseEvents`, `releaseTrustedSigners`, `releasePolicy`, and
`persistReleaseCheckpoint`. The low-level root API receives these inside
`session`; the application facade receives them directly. Policy requires:

```js
const releasePolicy = {
  now: verificationTimeUtc,
  minimumSequence: requiredSequence,
  checkpoint: persistedCheckpoint, // { sequence: 0, digest: null } only on first use
};
```

The application owns the trust anchor, reliable time, authorized release stream,
and durable monotonic checkpoint. Persist it atomically before execution; compare
and reject older or conflicting writes when multiple opens share that store.
Do not reset the checkpoint to bypass stale history. The verifier requires the
complete contiguous signed history starting at sequence 1, including rotations
authorized by the previous key. It rejects gaps, forks against the checkpoint,
old sequences, expired managed eligibility, and an event for another envelope. A later
signed rollback event can authorize an older executable; replaying an old event
cannot. A revoked executable root cannot be reactivated by a later event.
Offline verification cannot discover events it was never supplied: freshness,
minimum sequence, expiry, and distribution of revocations remain host duties.

The runtime also persists the verified checkpoint when an authenticated history
denies opening (including revocation, quarantine, expiry, or a different exact
envelope). A failed checkpoint write keeps execution denied and preserves both
errors. Invalid signatures, gaps, and checkpoint conflicts never advance state.
Consumers calling `verifyPack()` directly must likewise retain the checkpoint
on `PackReleaseStateError`; verification alone does not own storage.

### Recipient-controlled retained local use

Managed deployment remains the default. An application may separately retain an
explicit local-use decision while a release is eligible. Store it with the
already persisted release checkpoint and exact Pack bytes:

```js
const retainedLocalUse = {
  schema: 'doppler.pack-retained-local-use/v1',
  pack: { schema, semanticRoot, envelopeDigest },
  releaseEventDigest: acceptedEvent.digest,
  applicationDigest, // canonical SHA-256 of acceptedEvent.release.application
  acceptedAtUtc, // within this event's eligibility window
  acknowledgeUnseenRevocations: true,
};
// A later local open, explicitly chosen by the application:
const retainedPolicy = { ...releasePolicy, retainedLocalUse };
```

This is a recipient decision, not a publisher signature, trusted timestamp, or
proof of outside adoption. Doppler does not create it automatically, infer it
from cached bytes, or select a signer for the recipient. The caller supplies a
reliable verification time and the actual durable checkpoint on every open.

The verifier still requires the complete signed history, exact accepted event,
Pack envelope, application contract, artifact closure, qualified TargetPlan,
and successful checkpoint persistence. Known blocked, quarantined, revoked, or
superseded states deny execution even after expiry. A new event requires a new
explicit acceptance; a revoked root cannot be revived. Resetting durable state
or hiding newly received history is not supported rollback behavior.

`session.verification.lifecycle.authorization` records the mode, verification
time, whether managed eligibility expired, and the retained decision. Pack v3
operation receipts carry that same `releaseAuthorization` in their digest.
Generation's direct streaming interface retains its token contract; inspect
the session or use `executeOperation()` for its bound receipt. These are
opening-time observations, not a live revocation watcher or hardware attestation.

Retained mode refuses declared delegated assignments, including operation jobs,
sequence assignments, and forecast assignment hashes. It does not authorize
redistribution, resource contribution, or relaxed input acceptance. The signed
managed `release.revocation` policy is preserved, not rewritten: an expired
event remains expired. An external evaluator must inspect `releaseAuthorization`
rather than mistaking a local-use receipt for current managed eligibility.
No offline implementation can discover an unseen revocation. Applications that
require that assurance must keep managed fail-closed opening and fresh history.
This option requires v3; it is rejected on v2 rather than silently ignored.

## Execution evidence and limits

Pack opening filters by device capability, host qualification, optional
`acceptedTargetPlanDigests`, and every operation in optional `requiredOperations`
before selecting a plan. An omitted digest allowlist accepts any otherwise
qualified plan; an empty list accepts none. `preferredTargetPlanDigests` orders
eligible plans only and never grants authorization. Signed Pack order breaks
ties and remains the default when no preference is supplied. None of these
options rewrites a plan or switches an already-open session.

For example, `openPack(pack, { requiredOperations: ['rerank'],
acceptedTargetPlanDigests: [approvedDigest] })` on the host API selects an
approved reranking plan even when a generation-only plan appears first. The
injected root API places the same policy under `options.session`. The Electron
reranking adapter adds `rerank` without discarding caller-required operations.
Selection policy is copied before asynchronous opening, and no eligible plan
means no program loading. Per-operation qualification still applies at execution.

`encodeSequence` returns a `doppler.pack-execution-receipt/v1`
bound to the exact Pack, selected TargetPlan, full artifact receipts, release
event, assignment, input options, and semantic output. Timings are observations,
not semantic output identity. The operation supports cancellation and rejects
closed sessions. A receipt is an execution declaration, not hardware attestation.

Generation and reranking retain their existing contracts. This migration does
not claim new host support, throughput, ESM qualification, origin-independent
delivery, or completion of generic embedding/API convergence. Tests with injected
programs prove contracts; real application and device qualification remain
separate. Run `npm run pack-v2:check` for v2/v3 regression coverage and
`node tools/run-node-tests.js tests/pack/pack-retained-local-use.test.js` for
retained-use, denied-assignment, and durable-denial regressions.
