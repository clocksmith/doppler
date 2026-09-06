# Capsule naming migration

Doppler 0.6.0 replaces the product name **Pack** with **Capsule**. This is a
breaking source, package, CLI, and signed-format change, not an alias layer.
The version in this checkout does not imply an npm publication.

## Consumer changes

| Previous contract | Current contract |
| --- | --- |
| `openPack()` | `openCapsule()` |
| `doppler-gpu/pack` | `doppler-gpu/capsule` |
| `Pack*` declarations | `Capsule*` declarations |
| `createFetchPackArtifactStore()` | `createFetchCapsuleArtifactStore()` |
| `pack`, `packId`, `packIdentity`, `packSemanticRoot` | `capsule`, `capsuleId`, `capsuleIdentity`, `capsuleSemanticRoot` |
| `packSource`, `packTrustedSigners` | `capsuleSource`, `capsuleTrustedSigners` |
| `weightPackId`, `weightPackHash` | `weightCapsuleId`, `weightCapsuleHash` |
| `doppler.pack/v2`, `doppler.pack/v3` | `doppler.capsule/v2`, `doppler.capsule/v3` |
| `--pack-trusted-signers`, `--source-pack` | `--capsule-trusted-signers`, `--source-capsule` |
| `forge:pack`, `pack-v2:check` | `forge:capsule`, `capsule-v2:check` |

The root, `/runtime`, and `/host` exports retain their locations but expose
Capsule names. Former product-name modules, methods, and package subpaths are
removed. Update imports, declarations, request objects, receipt consumers,
configuration files, and scripts together. The application-bound reranking
request, cancellation, loading progress, and cleanup contracts remain intact.

## Artifact changes are not text substitution

Schemas, identity fields, signatures, and executable roots bind exact content.
Never rename strings inside a signed artifact and call it verified. Rebuild from
pinned source with Forge, sign the resulting Capsule with an explicitly chosen
authority, qualify it, and have the application approve its new identity.
Capsule v2-to-v3 migration accepts Capsule v2, not the former Pack format.

Preserve old release checkpoints and known denials. A naming change does not
authorize clearing revocations, resetting rollback protection, transferring
trust, or automatically activating a new release. Existing deployments may
retain their old pinned runtime independently; the new runtime does not load
their former-format artifacts.

Weight and tokenizer bytes are not product terminology. Reuse unchanged bytes
where the new descriptors and conversion contract permit it; manifests and
references still need correct new identities. No inference, hardware,
performance, or adoption claim follows merely from the rename.

## Historical evidence

Retained reports, benchmark observations, model files, tokenizers, and frozen
experiment inputs keep their original bytes and terminology. They document
their original runtime and format, not current Capsule support.
`tests/fixtures/pre-capsule/` holds exact old configuration inputs needed to
check those records after current configuration names change. It is not an
execution path, compatibility implementation, or public API.

New deterministic migration receipts live in `reports/capsule-migration/`.
They do not replace old reports or imply physical execution. The legacy HTTP
receipt inspector may read historical fields solely to validate those records;
the Capsule session contract does not accept them.

Ordinary terms such as `package.json`, `npm pack`, GPU bit packing, packed
tensor layouts, and vocabulary tokens are unchanged. Adjacent repositories
are not migrated here; their imports and receipt consumers must be updated
before they can use Doppler 0.6.0.
