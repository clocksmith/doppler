# Bun Product Qualification

Bun remains experimental until one portfolio-wide qualification promotes
generation, embedding, and reranking together. Historical Bun benchmark runs
are useful diagnostics but cannot establish this product claim.

## Authority

`tools/policies/bun-product-qualification.json` is the status authority.
`npm run bun:qualification:check` validates the three workload candidates,
their retained evidence, the Bun support subsystem, and the `doppler-bun`
release target. An evidence-complete candidate is still non-claimable until an
explicit promotion receipt exists.

Each workload binds its logical model and named manifest variant to the
runtime-observed manifest and execution SHA-256 identities in a local
`doppler_provider_receipt_v1`. Every additional receipt binds those identities
plus the exact Bun version, WebGPU implementation, provider, harness revision,
and environment fingerprint.

The required semantic evidence classes are:

- root API and CLI conformance with output parity and no hidden fallback;
- load, execute, unload, and repeated-session lifecycle;
- workload-appropriate correctness and held-out quality;
- crash, OOM, and device-loss reliability;
- peak-memory budget;
- cold and warm response distributions with sample sufficiency;
- a Bun-native incumbent comparison or a digest-bound no-eligible-incumbent finding;
- upgrade requalification with preserved identity and task quality;
- rollback plus observed revocation propagation.

Each policy reference includes a canonical JSON digest. The checker recomputes
every class result from its observations, rejects mixed host or execution
identities, requires evidence to predate qualification, and rejects reused
paths. Mocked or static evidence cannot be presented as hardware qualification.

## Recording

`npm run bun:qualification:record -- --capture <capture.json> --out
<candidate-policy.json>` reads one execution receipt and the nine semantic
receipt paths. It derives all digests, resolved identities, and host identity.
The capture cannot provide those values or a pass flag.

The recorder never promotes support. Its output keeps `claimAllowed: false`,
leaves `promotion` null, and adds an explicit promotion blocker. `--apply` is
required to change the configured policy, and `--replace` is required to replace
an existing non-claimable evaluation.

## Promotion

A claimable workload requires a digest-bound
`doppler.bun-product-promotion-evidence/v1` receipt with human authority. That
receipt binds the complete non-promotion evidence set, exact identities,
qualification time, and expiry.

Partial promotion is forbidden. All three workload receipts, the
`runtime.bun-webgpu` tier change from `experimental` to `tier1`, and the
canonical `doppler-bun` vendor-registry change from `experimental` to `active`
must land as one coherent promotion; the generated release matrix must reflect
the same status. If any workload expires or loses evidence, the broad Bun
product claim no longer satisfies the gate.
