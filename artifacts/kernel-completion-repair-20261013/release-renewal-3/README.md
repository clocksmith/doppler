# Test release renewal, sequence 3

Component: signed model delivery (local qualification scope). Intent preserved.
Boundary effects: one new release event in the test authority's history; no Capsule,
archive, catalog, installed package or public pin changed.

The candidate Capsule (semantic root `67f240af...`, runtime archive `2df37d88...`) was
eligible through sequence 2 (`923ac7e6...`), which expires 2026-10-15T02:14:39Z and fails
closed. [`renew.js`](renew.js) appends sequence 3 before that expiry:

- the same Capsule envelope, release contract and migration source as sequence 2;
- `previousEventDigest` is sequence 2's digest;
- action `eligible`, 24-hour expiry from issue, signed by the test authority
  `reploid-paired-qualification-20261012` (key read from its file, never printed).

[`release-events.json`](release-events.json) is the full renewed history (sequences 1 to 3).
[`renewal-verification.json`](renewal-verification.json) records `verifyCapsule` from the
exact 0.6.28 archive admitting it from checkpoints at sequence 2 (the final-5 installs),
1 and 0: every path reaches checkpoint 3 (`74ab985f...`, expires 2026-10-15T12:32:31Z).

Delivery: final-5's catalog (`chat-models.json` `packOpenOptions.releaseEvents`) still carries
sequences 1 and 2. Hosts admit the renewal only through a delivery that carries this history.
A later Capsule from this authority continues at sequence 4.

Scope: test authority only. Production signing remains the public-promotion gate.
