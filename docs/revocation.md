# Model, Artifact, and Adapter Revocation

`src/config/revocation-registry.json` is the shipped authority for model,
artifact, and adapter revocation. It can target exact logical model IDs,
resolved model IDs, source checkpoints, weight packs, manifest variants,
manifest-byte SHA-256 identities, adapter IDs, and adapter SHA-256 identities.
An adapter digest may identify verified source weights or the canonical loaded
tensor-layout execution identity; records should include the adapter ID and
every known digest that must be denied.

The registry is deny-only. A record cannot expire or silently restore an older
artifact. It may name replacements for operators, but runtime resolution never
selects one automatically.

## Runtime enforcement

The root API rejects matching quickstart, explicit-URL, inline-manifest, cached,
and weights-ref loads. Known identity is checked during source resolution, then
the complete manifest identity is checked before device initialization and
weight loading. The exported legacy model manager checks the requested model
before loading and the manifest-owned source, weight-pack, and variant IDs
before pipeline construction.

The LoRA loader checks a manifest ID and declared source checksum before it
fetches weights, then checks the observed source digest and canonical execution
digest before authorizing the loaded object. The text pipeline accepts only an
object authorized by that path, so direct unchecked `setLoRAAdapter()` calls
fail closed. Unload remains permitted.

Malformed registries fail closed. `DopplerRevocationError` exposes the record
ID, severity, matched identities, and named replacements without authorizing a
fallback.

## Propagation contract

Issuing a revocation requires the same change to:

1. add the immutable record and retained evidence;
2. set matching catalog entries to `lifecycle.status.runtime="revoked"`, with
   `quickstart=false` and `demoVisible=false`;
3. set matching adapter-catalog entries to `lifecycle="revoked"`;
4. withdraw matching claim, integration, provider, and runtime-ownership rows;
5. run `npm run quickstart:sync` and regenerate affected reports;
6. run `npm run revocations:check`.

The checker rejects revoked quickstart entries, active model or adapter catalog
states, promoted claim lanes, claim-ready release rows, product integrations,
provider results, and runtime-ownership decisions. Historical failed and
rejected evidence stays retained; it does not remain claimable.

## Signed live update mechanism

The root API exposes an opt-in control plane:

- `dr.revocations.configure(options)` binds one exact HTTPS endpoint, trusted
  online and recovery P-256 public keys, resource limits, clock policy, and an
  application-owned durable state store, restores verified state, and performs
  an initial refresh;
- `dr.revocations.refresh({ force })` explicitly requests an update; Doppler
  performs no hidden background networking;
- `dr.revocations.status()` returns the installed epoch, sequence, expiry,
  signature state, current/stale result, and offline/error state.

The `doppler.signed-revocation-envelope/v1` signature covers canonical JSON for
the authority ID, epoch, sequence, issue and expiry instants, deny registry,
optional keyring transition, and signer ID. Verification uses WebCrypto ECDSA
P-256 with SHA-256. Online updates must retain the current epoch and advance the
sequence. A recovery update must be signed by a separately trusted recovery
key, advance the epoch, install the complete next online keyring, and retain all
previously revoked public-key identities. Any removed online key becomes a
retained revoked key, and trusted recovery key material cannot be activated as
an online key. Recovery keys rotate through package trust updates,
not a live online-key assertion.

Every accepted update must retain every prior live deny record byte-for-byte
after normalization. Exact replays are idempotent; rewritten replays, sequence
rollback, epoch rollback, expired state, excessive lifetime, future issuance,
oversized responses, redirects, and malformed trust pairs fail closed. Doppler
saves the verified envelope before installing it. A failed refresh may use
cached verified state only until its signed expiry.

The supplied state store is part of the trusted computing base. Its `save()`
must be atomic and durable. Sequence checks detect rollback relative to the
state returned by that store; they cannot detect wholesale rollback or deletion
of an untrusted store. Applications must also configure finite response-byte,
request-timeout, clock-skew, envelope-lifetime, and refresh-interval limits.

The Electron release adapter adds a separate application update-state boundary.
Its main-process coordinator uses an application-supplied atomic
`compareAndSwap()` store for current, previous, candidate, retained failure, and
verified revocation-snapshot state. Candidate installation never activates a
Pack. Activation requires a cryptographically verified eligible decision plus
an explicit customer authorization digest; rollback also requires an explicit
customer authorization digest. A revocation snapshot carries its policy digest,
issuance and expiry instants, monotonic sequence, revoked Pack semantic roots,
content digest, and signature. The coordinator calls an application-owned
cryptographic verifier before committing it; a caller-authored boolean is not
verification. Renderer execution fails closed when the signed snapshot is
missing, expired, exceeds the activated release's offline-expiry window, binds
a different authority or policy, or denies the current Pack. This is a
repository contract and reference implementation, not evidence that a customer
has deployed an atomic store or exercised production rollback.

Installing a new live policy increments the process policy revision. Already
loaded model identity is checked again at every transformer layer boundary, and
an active adapter is re-authorized against the new revision before layer work.
A newly denied identity therefore cannot silently continue into the next layer;
already submitted GPU work is not retroactively cancelled. Unload remains
permitted, and named replacements are never selected automatically.

## Current qualification boundary

The bundled registry remains the default authority trusted through the
installed package. Doppler now provides the signed update, rotation, recovery,
offline, persistence, and loaded-identity enforcement mechanism, but the
package does not configure a production endpoint or production keys. No live
authority is claimable until Clocksmith deploys and qualifies the endpoint,
online and recovery key custody, a production durable store, refresh behavior,
and retained failure and recovery drills.

`tools/policies/signed-revocation-authority-qualification.json` makes this
boundary executable. A claimable authority must bind one exact HTTPS endpoint
with redirects forbidden, an authority ID, disjoint online and recovery key
IDs, browser and Node durable-state store IDs, a current named owner, and a
finite qualification expiry. It must retain repo-visible evidence for endpoint
deployment, package trust, independent custody, both stores, current refresh,
key rotation, exact and rewritten replay behavior, sequence and epoch rollback,
offline expiry, compromise recovery, durable restart, loaded-identity
invalidation, application fail-closed behavior, and requalification.
The policy separately bounds owner-confirmation and operational-evidence age;
qualification expiry cannot outlive either freshness window.

Every evidence entry is null or a repository-relative `{ path, digest }`
reference to canonical JSON. Ownership uses
`doppler.signed-revocation-authority-owner-confirmation/v1`; the remaining
classes use `doppler.signed-revocation-authority-evidence/v1` and bind the
qualification, owner, production authority ID, harness revision, environment,
capture time, and class-specific observations. The checker derives each pass
from those observations, cross-checks endpoint, key, and durable-store
identities, requires all operational receipts to share one exact harness
revision and environment fingerprint, rejects reused evidence paths, and
retains failed drills as named qualification reasons. A document path or
operator-entered `passed` value is not production-authority proof.

Record a retained evaluation with `npm run revocations:authority:record --
--capture <capture.json> --out <candidate-policy.json>`. The capture contains
receipt paths and evaluation bounds, not copied digests or deployment facts.
The recorder derives canonical digests, ownership time, endpoint, authority ID,
key identities, durable-store identities, and semantic failure blockers. It
writes a separate candidate by default; `--apply` is required for the status
authority and `--replace` for prior evaluation state. It always forces
`lifecycle: candidate` and `claimAllowed: false`. Production activation remains
a separate human authority decision. The recorder clears promotion evidence and
cannot replace a promoted authority.

Production activation requires
`doppler.signed-revocation-authority-promotion-evidence/v1`. Its human reviewer
binds the endpoint, authority, exact key and durable-store identities, shared
harness/environment identity, qualification and expiry times, and canonical
digest of every pre-promotion evidence reference. The decision must be
`promote-production-authority`. Changing any retained reference invalidates the
promotion; editing lifecycle, blockers, or `claimAllowed` cannot create a
production authority.

Run `npm run revocations:authority:check` to validate that contract. The current
entry is only a candidate: its production endpoint, keys, stores, and receipts
remain null, its blockers are explicit, and `claimAllowed` remains false. The
goal matrix therefore keeps signed live revocation partial. Product readiness
reports bundled signature state, mechanism availability, contract validity,
candidate count, and qualified authority count as distinct states.
