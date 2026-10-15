# Test release renewal, sequence 10

Component: signed model delivery (local qualification scope). Intent preserved.
Boundary effects: one new release event in the test authority's history; no Capsule,
archive, catalog, installed package or public pin changed.

Sequence 9 (`dd837996...`, Capsule 3 of the [faster candidate](../README.md): manifest
`548e6559...`, semantic root `b1527013...`, runtime archive `3b132ee7...`) expires
2026-10-16T04:35:38Z and fails closed. [`renew.js`](renew.js) appends sequence 10 before that
expiry, using the procedure of [renewal 6](../../chunked-prefill-capsule-20261014/release-renewal-6/README.md):

- the same Capsule envelope, release contract and migration source as sequence 9;
- `previousEventDigest` is sequence 9's digest;
- action `eligible`, 24-hour expiry from issue, signed by the test authority
  `reploid-paired-qualification-20261012` (key read from its file, never printed).

[`release-events.json`](release-events.json) is the full renewed history (sequences 1 to 10).
[`renewal-verification.json`](renewal-verification.json) records `verifyCapsule` from the
candidate runtime admitting it from checkpoints 9 (final-10 pass 3), 8, 7, 6, 5 (final-9) and 0.
Every path reaches checkpoint 10 (`e7b0cb34...`, expires 2026-10-16T13:18:27Z).

Delivery: hosts admit the renewal only through a delivery whose catalog
(`chat-models.json` `packOpenOptions.releaseEvents`) carries this history. Final-10 pass 3
carries sequence 9.

Scope: test authority only. Production signing remains the public-promotion gate.
