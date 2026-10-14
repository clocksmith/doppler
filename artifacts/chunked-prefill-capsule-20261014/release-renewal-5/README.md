# Test release renewal, sequence 5

Component: signed model delivery (local qualification scope). Intent preserved.
Boundary effects: one new release event in the test authority's history; no Capsule,
archive, catalog, installed package or public pin changed.

The chunked-prefill Capsule (semantic root `171132d4...`, runtime archive `2df37d88...`) was
eligible through sequence 4 (`7f62c19a...`), which expires 2026-10-15T13:21:16Z and fails
closed. [`renew.js`](renew.js) appends sequence 5 before that expiry:

- the same Capsule envelope, release contract and migration source as sequence 4;
- `previousEventDigest` is sequence 4's digest;
- action `eligible`, 24-hour expiry from issue, signed by the test authority
  `reploid-paired-qualification-20261012` (key read from its file, never printed).

[`release-events.json`](release-events.json) is the full renewed history (sequences 1 to 5).
[`renewal-verification.json`](renewal-verification.json) records `verifyCapsule` from the
exact 0.6.28 archive admitting it from checkpoints at sequence 4, 3 and 0: every path reaches
checkpoint 5 (`70f24c97...`, expires 2026-10-15T16:14:00Z).

Delivery: hosts admit the renewal only through a delivery whose catalog
(`chat-models.json` `packOpenOptions.releaseEvents`) carries this history.

Scope: test authority only. Production signing remains the public-promotion gate.
