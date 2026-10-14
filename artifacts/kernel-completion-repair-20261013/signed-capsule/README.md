# Isolated signed Capsule for candidate 2df37d88

Component: signed model delivery (local qualification scope). Intent preserved.
Boundary effects: a new isolated Capsule and release history only; no catalog,
application package, running contributor, public pin or deployment changed.

`build.js` binds the three-stage qualified candidate into a signed Capsule. It
follows the earlier two-stage build
(`../../qwen35-08b-capsule-intake-20261012/signed-closure-candidate/build.js`)
step for step with this candidate's inputs:

- runtime archive `2df37d88…` and an exact source export at `a5f3ca4c`;
- model manifest `6b2cb026…`, with weight and tokenizer pieces asserted equal
  to the previous signed piece index;
- the three partition reports and v2 contract from Reploid
  `artifacts/completion-qualification-2-20261013/reports/`;
- a generation report derived from that campaign's capture, with exact
  source-token parity for the first request.

The program bundle reproduces execution-graph digest `sha256:9e28445d…`, the
digest the reports were qualified against. `build-binding.json` records every
input hash and the Capsule identity; `exact-runtime-verification.json` records
`verifyCapsule` from the exact installed archive accepting all artifacts.

## Signing scope

The signer is the isolated test authority `reploid-paired-qualification-20261012`.
Its policy scope is local signed browser/Node opening qualification only, with
no public eligibility or application promotion. `trusted-signers.json` holds
the public key; the private key is not in this repository.

## Release history

Release checkpoints are keyed by authority and application, so every Capsule
from this authority shares one monotonic history. A contributor that accepted
the earlier `0fad` release holds sequence 1 with that event's digest, and a
second sequence-1 history would be rejected as a fork. `release-events.json`
therefore carries that earlier event followed by this Capsule's event at
sequence 2, chained by digest. The build verifies the history both for a host
at that checkpoint and for a host with no history. The new event expires 24
hours after issue and fails closed. `release-policy.json` names this
candidate's model identity; the inherited policy named an older manifest.

## Hosts

Hosts are named `mac-host`, `amd-host` and `intel-host`. Two earlier builds
from the same inputs are superseded and not retained here: the first inherited
a private machine address from an older generation report into a signed
artifact, and the second started a forked sequence-1 history and labelled the
Mac by its local name. Neither was delivered. `build.js` now sets aliases and
refuses to finish if an address is present.

## Not established

Physical signed opening, the signed catalog and staged application packages,
installed conversations, and any public eligibility.
