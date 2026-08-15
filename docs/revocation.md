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
and retained rotation and compromise-recovery drills. The goal matrix therefore
keeps signed live revocation partial and `claimAllowed: false`. The product
readiness report exposes bundled signature state and signed-live mechanism and
authority state separately so this boundary remains observable.
