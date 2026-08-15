# Model and Artifact Revocation

`src/config/revocation-registry.json` is the shipped authority for model and
artifact revocation. It can target exact logical model IDs, resolved model IDs,
source checkpoints, weight packs, manifest variants, and manifest-byte SHA-256
identities.

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

Malformed registries fail closed. `DopplerRevocationError` exposes the record
ID, severity, matched identities, and named replacements without authorizing a
fallback.

## Propagation contract

Issuing a revocation requires the same change to:

1. add the immutable record and retained evidence;
2. set matching catalog entries to `lifecycle.status.runtime="revoked"`, with
   `quickstart=false` and `demoVisible=false`;
3. withdraw matching claim, integration, provider, and runtime-ownership rows;
4. run `npm run quickstart:sync` and regenerate affected reports;
5. run `npm run revocations:check`.

The checker rejects revoked quickstart entries, active catalog states, promoted
claim lanes, claim-ready release rows, product integrations, provider results,
and runtime-ownership decisions. Historical failed and rejected evidence stays
retained; it does not remain claimable.

## Current trust boundary

The registry is trusted because it ships inside the installed Doppler package.
It is not fetched dynamically and cryptographic signature verification is
currently unavailable. Therefore a newly issued record reaches an application
only through a package update and process restart. Doppler must not describe
this as live or signed revocation until an authenticated update channel,
anti-rollback state, offline behavior, key rotation, and compromise recovery
are implemented and qualified.
