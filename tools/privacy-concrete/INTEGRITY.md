# Malicious-evaluator threat model and the checked-integrity route

Status: research. Production cryptographic admission remains closed. Nothing
here has been reviewed by an independent cryptographer; the [review record](#review-record)
lists what has been examined and by whom. The full-replay routes and their
retained receipts are unchanged and remain the reference integrity mechanism.

This document owns the adversary model for a malicious evaluator and the
argument for `secret-linear-check-sampled-replay/v1`, the mechanism the
`qwen-all-linear-ntt12-check` profile uses instead of replaying every encrypted
tile on the client. The [security boundary](SECURITY-BOUNDARY.md) still owns the
parameter, deployment and leakage gates that both mechanisms share.

## Setting

The client holds the secret key, the prompt, every activation, the caches and
the output. For each linear projection it quantizes its activation to an integer
row `x` (each entry in `[0, 2^12)`), encrypts it and sends it. The evaluator holds
only public material: the integer coefficient tile `W` (entries in
`[-2^11, 2^11)`), the evaluation key and ciphertexts. It returns a ciphertext
that an honest evaluation decrypts to `y = xW + e`. `e` is honest decode error:
packing key switch and compression drop low-order bits, so even an honest
response is not exactly `xW`.

Two facts shape everything below. The client knows `x` and `W` in the clear, so
it can state exactly what the response should decode to. And `e` is not zero, so
no check on decoded values can demand equality.

## Adversary

The malicious evaluator controls every evaluator host and the transport behind
the authenticated endpoint. All remote operators may collude. It:

- sees every frame, length, order and inter-frame timing;
- returns arbitrary bytes for any response, chosen adaptively from everything it
  has seen in this and earlier attempts;
- can replay any response it produced earlier, in this attempt or another;
- can compute any function of the ciphertexts that the public evaluation key
  permits, including valid evaluations of a different matrix;
- can observe, outside the protocol, whether the requester later starts another
  attempt.

It cannot read client memory, cannot break the GLWE encryption (assumption A2)
and cannot predict the client's attempt-local randomness.

Out of scope, unchanged from the security boundary: a compromised client host,
timing and length side channels, availability (the evaluator can always refuse
to answer), model confidentiality (weights are public) and the unreviewed
cryptographic parameters.

## Goals

- **G1, input confidentiality.** The evaluator learns nothing about `x` beyond
  the leakage already listed in the security boundary, plus the explicitly
  bounded leak in Claim 4.
- **G2, bounded-deviation integrity.** If the client releases an output, every
  response that fed it decoded to within a stated distance of `xW`, except with
  stated probability. This is weaker than the equality that full replay gives.
- **G3, no silent downgrade.** The checked route is a separate installed profile,
  plan schema and protocol identifier. Neither side can substitute it for the
  replay route or the reverse.

Not a goal: exact-token integrity. Honest decode error already moves logits
(0.12 to 0.16 on the retained small-Qwen runs). A response inside the accepted
distance can move them too.

## Mechanism

Per attempt the client draws a fresh 256-bit check secret. It is never sent,
never written to a plan, report or cache name, and is discarded at close.

1. **Check vectors.** For each tile the client derives `t = 16` secret integer
   vectors `r_1..r_t`, each entry uniform in `[-2^15, 2^15)`, from
   SHAKE-256(secret, tile identity). It computes `u_k = W r_k` once, exactly.
2. **Length gate.** A response whose length differs from the trusted local
   length for its geometry is rejected without being parsed. The first response
   of each geometry is always replayed, which is where the trusted length comes
   from.
3. **Linear check.** For every other response the client decrypts, then accepts
   the row iff `|<r_k, y> - <x, u_k>| <= tau(m)` for all `k`, where `m` is the
   tile's output width. All arithmetic is exact in 64-bit integers; the code
   rejects geometries that could overflow. Decoded values outside the signed
   35-bit domain fail.
4. **Sampled exact replay.** With probability 1/16 per tile and step, decided by
   HMAC-SHA256(secret, tile identity, step), the client also runs the
   deterministic encrypted evaluation itself and compares bytes, exactly as the
   replay route does for every tile.
5. **Deferred rejection.** A failed gate, parse, decryption, check or replay is
   recorded client-locally. The client replaces that tile's contribution with
   zeros, keeps executing the fixed schedule, finishes and closes the transport,
   and only then raises `integrity-check-failed` and releases nothing.

Cancellation, deadline, transport and admission failures still stop
immediately and terminally, as before.

### Tolerance

`tau(m) = ceil(2^15 * B(m) * z)`, with `z = sqrt(2 ln 2 * 41) = 7.54` and
`B(m) = margin * sqrt(80 + 7.5 m) * (sqrt(m) + z)`, `margin = 2`.

`B(m)` is the assumed bound on the Euclidean norm of an honest tile's decode
error (assumption A1). The fit and the margin come from the
[measured sweep](evidence/integrity/honest-noise-ntt12.json): per-coordinate
variance is about `7.4 m` on uniform inputs, and one extreme-input case showed a
data-dependent bias that raised the largest row norm to 8,791 at `m = 2048`,
against `B(2048) = 13,120`.

## Claims

Write `delta = y - xW` for the deviation of a decoded row and `R = 2^15`.

**Claim 1 (linear check soundness).** Fix a tile and a row. For any response
chosen without knowledge of the check secret, if some coordinate of `delta` has
magnitude `D > 0`, all `t` checks accept with probability at most
`((2 tau / D + 1) / 2R)^t`.

*Argument.* `<r_k, y> - <x, u_k> = <r_k, delta>` exactly, because `u_k = W r_k`
and the arithmetic does not overflow. Let `j` be a coordinate with
`|delta_j| = D`. Condition on every other entry of `r_k`. Then
`<r_k, delta> = a + r_kj delta_j` for a fixed `a`, and `r_kj` is uniform over
`2R` consecutive integers, so the value ranges over `2R` points spaced `D`
apart. At most `2 tau / D + 1` of them lie in `[-tau, tau]`. The `t` vectors are
independent. The response is independent of the secret because nothing derived
from the secret is ever sent and, within an attempt, no accept or reject
outcome changes anything the evaluator can see (Claim 4). ∎

The bound is vacuous for small `D`. Deviation that every check accepts with
probability at most `2^-40`, and at most `2^-16`:

| Output width `m` | `tau / R` | `D` for `2^-40` | `D` for `2^-16` |
| --- | --- | --- | --- |
| 128 | 9,167 | 51,863 | 18,336 |
| 640 | 34,588 | 195,678 | 69,179 |
| 896 | 46,592 | 263,590 | 93,188 |
| 2048 | 98,914 | 559,590 | 197,834 |

For scale, on uniform twelve-bit data the exact products have standard deviation
about `1.25e8`, so the `2^-40` threshold at `m = 2048` is about 0.45% of that.
Real activations are smaller, so the relative threshold is larger; it has not
been measured on a model.

**Claim 2 (no false rejection of honest responses).** If an honest tile's
decode error satisfies `||e|| <= B(m)` (A1), a single check rejects with
probability at most `2 exp(-z^2 / 2) = 2^-40`.

*Argument.* For fixed `e`, `<r_k, e>` is a sum of independent terms bounded by
`R |e_j|`. Hoeffding gives `P(|<r_k, e>| > tau) <= 2 exp(-tau^2 / (2 R^2 ||e||^2))`. ∎

A1 is an empirical assumption about this native build and parameter profile,
not a theorem. If it fails, honest attempts are rejected; soundness is
unaffected.

**Claim 3 (sampled replay).** If the evaluator returns anything other than the
exact honest bytes for `k` tile-steps, at least one is caught with probability
`1 - (15/16)^k`, whatever the size of the change, provided HMAC-SHA256 is a
pseudorandom function (A3).

*Argument.* Each tile-step is sampled independently by a keyed function the
evaluator cannot evaluate. Replay compares against the deterministic honest
ciphertext, so it does not depend on the plaintext or on the size of the change.
The evaluator may learn afterwards that a tile was sampled, because the client's
next frame is delayed by its own replay. That is after it has committed to the
response, and the sample bits are independent. ∎

One tampered tile-step survives with probability 15/16. Sampled replay deters
sustained small tampering; it does not prevent a single targeted change inside
the Claim 1 tolerance.

**Claim 4 (what rejection reveals).** Within an attempt the evaluator's view is
the same whether or not any integrity check failed, up to the encryptions
themselves (A2) and client timing. Across attempts the evaluator can learn at
most one bit per attempt: whether the attempt was rejected, and only if it can
observe the requester's later behaviour.

*Argument.* Failures are recorded locally. The client sends the same number of
frames with the same kinds, identities and lengths, then the same finish. The
focused test compares the complete wire trace of an honest and a tampered
attempt. Plaintexts differ after a zero substitution, which A2 hides. ∎

That bit is adversary-chosen. An evaluator can return a response that is correct
only when a linear function of `x` is near zero, so retrying the same private
request with an evaluator that caused a rejection leaks one chosen predicate per
retry. **Operating rule: an integrity rejection marks that operator as hostile.
Do not retry the same private input with it.** The research driver has no
automatic retry.

With immediate abort the position of the first failing tile would be visible,
which is about 17 bits per attempt on the Flash route. Deferral removes that.

**Claim 5 (rejected content does not feed forward).** After a rejected
response, no later client plaintext depends on that response's content.

*Argument.* The tile's contribution is replaced by zeros before accumulation.
The unparsed or decrypted value is dropped. ∎

## What this gives up relative to full replay

| Property | Full replay | Checked route |
| --- | --- | --- |
| Accepted response | Exact honest bytes | Decodes within `tau`-bounded distance of `xW`; exact only on the sample |
| Client parses unverified bytes | Never | Yes, after the length gate |
| Client decrypts unverified ciphertexts | Never | Yes |
| Rejection depends on private data | No | Yes; deferred, so at most one bit per attempt |
| Client encrypted work | Every tile | First of each geometry plus 1/16 of tiles |

Consequences that are stated, not solved:

- **Noise-sized steering.** A deviation inside the tolerance on every tile is
  accepted by the linear check. The evaluator chooses its direction; honest
  error is random. Whether a model's output can be steered this way, and how
  many tiles it takes, is a property of the model and is not established.
  Claim 3 is the only defence, and it is probabilistic.
- **Key-dependent plaintexts (A4).** An accepted response can make a later
  client plaintext depend on the secret key to within the tolerance. The
  evaluator sees only encryptions of it. This relies on the encryption remaining
  secure for such messages, an assumption the replay route did not need.
- **Parser exposure.** The native deserializer panics on malformed input and
  trusts size fields inside a correct-length response; one corrupted field made
  it expect a 201 MB buffer before failing an assertion. Panics are contained
  and recorded as rejections. No memory-safety or resource-bound claim is made
  for the parser.
- **No decryption oracle within an attempt, one bit across attempts.** Keys are
  fresh per attempt, so the evaluator gets no repeated reaction queries against
  one key. This is an argument, not a proof of security against chosen
  ciphertexts.

## Cost

The client still does, per attempt: one exact product `W r` per tile (16 integer
vectors), one decryption and 16 inner products per tile row, and an encrypted
replay of the first response of each geometry plus a 1/16 sample. It no longer
replays the remaining tiles.

The check needs the client to hold `W`. A client that holds `x` and `W` could
also compute `xW` itself with no evaluator. The checked route makes the client's
integrity work much cheaper than replay; it does not make the hybrid route
cheaper than local clear execution. Measured costs are in
[FULL-QWEN.md](FULL-QWEN.md#checked-integrity-route).

## Tests

| Concern | Test |
| --- | --- |
| Soundness bound, completeness, exact arithmetic, domain guards | `test_linear_check.py` |
| Deferred rejection with an unchanged wire trace, zero substitution, cancellation, public-weight change, earlier-step replay, secret never on the wire | `test_checked_bridge.py` |
| Real ciphertexts: wrong matrix, wrong input, wrong key, wrong length, bit flips, the noise-sized gap, sampled replay | `test_checked_native.py` (opt-in) |

The native bit-flip test flips 464 distinct bits of a real response. Every
accepted flip decoded within the Claim 1 threshold; the rest were rejected.

## Assumptions

- **A1.** Honest decode error of a tile has Euclidean norm at most `B(m)`.
  Measured on synthetic tiles only.
- **A2.** The GLWE encryption hides plaintexts from the evaluator under the
  installed parameters. Unreviewed; see the security boundary.
- **A3.** HMAC-SHA256 and SHAKE-256 keyed with the check secret are
  pseudorandom, and the operating system's random source is unpredictable.
- **A4.** Encryption stays secure when later plaintexts depend on the key
  through an accepted response.

## Open before any admission

Independent cryptographic review of this argument and of A2 and A4; a measured
A1 on real model activations for each admitted geometry; a model-level bound or
measurement for noise-sized steering; parser hardening or a verified
length-and-structure gate; a retry and operator-exclusion policy enforced in
code; timing analysis of the deferred path; and everything already listed in
the security boundary.

## Review record

No review has been recorded yet.
