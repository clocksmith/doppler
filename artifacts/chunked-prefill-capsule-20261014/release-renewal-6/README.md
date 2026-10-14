# Test release renewal, sequence 6

Component: signed model delivery (local qualification scope). Intent preserved.
Boundary effects: one new release event in the test authority's history; no Capsule,
archive, catalog, installed package or public pin changed.

Sequence 5 (`70f24c97...`, [renewal 5](../release-renewal-5/README.md)) expires
2026-10-15T16:14:00Z and fails closed. [`renew.js`](renew.js) appends sequence 6 before that
expiry, using the same procedure as renewal 5:

- the same Capsule envelope, release contract and migration source as sequence 5;
- `previousEventDigest` is sequence 5's digest;
- action `eligible`, 24-hour expiry from issue, signed by the test authority
  `reploid-paired-qualification-20261012` (key read from its file, never printed).

[`release-events.json`](release-events.json) is the full renewed history (sequences 1 to 6).
[`renewal-verification.json`](renewal-verification.json) records `verifyCapsule` from the
exact 0.6.28 archive admitting it from checkpoints at sequence 5 (`final-9`), 4 (`final-6` to
`final-8`) and 0. Every path reaches checkpoint 6 (`b19052d5...`, expires
2026-10-15T19:34:07Z).

Delivery: hosts admit the renewal only through a delivery whose catalog
(`chat-models.json` `packOpenOptions.releaseEvents`) carries this history. No such delivery
exists yet; `final-9` carries sequence 5.

Scope: test authority only. Production signing remains the public-promotion gate.
