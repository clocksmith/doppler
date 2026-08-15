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

## Current trust boundary

The registry is trusted because it ships inside the installed Doppler package.
It is not fetched dynamically and cryptographic signature verification is
currently unavailable. Therefore a newly issued record reaches an application
only through a package update and process restart. Doppler must not describe
this as live or signed revocation until an authenticated update channel,
anti-rollback state, offline behavior, key rotation, and compromise recovery
are implemented and qualified.
