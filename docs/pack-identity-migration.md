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
old sequences, expired eligibility, and an event for another envelope. A later
signed rollback event can authorize an older executable; replaying an old event
cannot. A revoked executable root cannot be reactivated by a later event.
Offline verification cannot discover events it was never supplied: freshness,
minimum sequence, expiry, and distribution of revocations remain host duties.

## Execution evidence and limits

`acceptedTargetPlanDigests` constrains selection; no accepted plan means no
program loading. `encodeSequence` returns a `doppler.pack-execution-receipt/v1`
bound to the exact Pack, selected TargetPlan, full artifact receipts, release
event, assignment, input options, and semantic output. Timings are observations,
not semantic output identity. The operation supports cancellation and rejects
closed sessions. A receipt is an execution declaration, not hardware attestation.

Generation and reranking retain their existing contracts. This migration does
not claim new host support, throughput, ESM qualification, origin-independent
delivery, or completion of generic embedding/API convergence. Tests with injected
programs prove contracts; real application and device qualification remain
separate. Run `npm run pack-v2:check` for v2/v3 regression coverage.
