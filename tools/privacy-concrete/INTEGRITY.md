# Malicious-evaluator threat model and the checked-integrity route

> Historical research record: see [Status](README.md#status) for the code removal and the 2026-11-17 stop; the commands for the removed scripts are kept in [this guide at commit 3e485ca93](https://github.com/clocksmith/doppler/blob/3e485ca9303ee51131884b6242ce4a7fe4753c24/tools/privacy-concrete/INTEGRITY.md).

Status: research. Production cryptographic admission remains closed. No
cryptographer has reviewed this. One independent automated review has been done
and its findings are applied or stated below; see the [review record](#review-record).
The full-replay routes and their retained receipts are unchanged and remain the
reference integrity mechanism.

This check belongs to the homomorphic route stopped on 2026-11-17 (see
[Status](README.md#status)); the mechanism and its small-Qwen evidence are retained.

This document owns the adversary model for a malicious evaluator and the
argument for `secret-linear-check-sampled-replay/v1`. Two separate installed
profiles use it instead of replaying every encrypted tile on the client:
`qwen-all-linear-ntt12-check` and `flash-text-all-linear-ntt12-check`. The
[security boundary](SECURITY-BOUNDARY.md) still owns the parameter, deployment
and leakage gates that both mechanisms share.

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

- sees every frame, length, order and inter-frame timing, including how fast the
  client reads;
- returns arbitrary bytes for any response, chosen adaptively from everything it
  has seen in this and earlier attempts;
- can replay any response it produced earlier, in this attempt or another;
- can compute any function of the ciphertexts that the public evaluation key
  permits, including valid evaluations of a different matrix;
- can end the route at any point by closing the connection or exiting;
- can observe, outside the protocol, whether the requester later starts another
  attempt.

It cannot read client memory, cannot break the GLWE encryption (assumption A2)
and cannot predict the client's attempt-local randomness.

Out of scope, unchanged from the security boundary: a compromised client host,
timing and length side channels other than those named here, availability,
model confidentiality (weights are public) and the unreviewed cryptographic
parameters.

## Goals

- **G1, input confidentiality.** The evaluator learns nothing about `x` beyond
  the leakage already listed in the security boundary, plus the explicitly
  bounded leak in Claim 4.
- **G2, bounded-deviation integrity.** If the client releases an output, every
  response that fed it decoded to within a stated distance of `xW`, except with
  stated probability. This is weaker than the equality that full replay gives.
- **G3, no silent downgrade.** The checked route has its own profiles, plan
  schemas, protocol identifiers and SSH route schema. The checked bridge refuses
  a replay-protocol plan and the replay bridge is never selected for a checked
  profile.

Not a goal: exact-token integrity. Honest decode error already moves logits
(0.12 to 0.16 on the retained small-Qwen runs). A response inside the accepted
region can move them further, in a direction the evaluator chooses.

## Mechanism

Per attempt the client draws a fresh 256-bit check secret. It is never sent,
never written to a plan, report or cache name, and is discarded at close.

1. **Check vectors.** For every tile, route step and input row the client
   derives `t = 16` secret integer vectors `r_1..r_t`, each entry uniform in
   `[-2^15, 2^15)`, from SHAKE-256(secret, tile identity, step, row count). No
   two rows or steps share vectors. It computes `u_k = W r_k` exactly.
2. **Length gate.** A response whose length differs from the trusted local
   length for its geometry is rejected without being parsed. The first response
   of each geometry is always replayed, which is where the trusted length comes
   from.
3. **Linear check.** For every other response the client decrypts, then accepts
   the row iff `|<r_k, y> - <x, u_k>| <= tau(m)` for all `k`, where `m` is the
   tile's output width. All arithmetic is exact in 64-bit integers; the code
   rejects geometries that could overflow. A decoded value outside the signed
   35-bit domain fails.
4. **Sampled exact replay.** With probability 1/16 per tile and step, decided by
   HMAC-SHA256(secret, tile identity, step), the client also runs the
   deterministic encrypted evaluation itself and compares bytes. It starts the
   replay only after the complete response is in client memory.
5. **Deferred rejection.** A failed gate, parse, decryption, check or replay is
   recorded client-locally. The client replaces that tile's contribution with
   zeros, keeps executing the fixed schedule, finishes and closes the transport,
   and only then raises `integrity-check-failed` and releases nothing. If the
   route ends any other way after a failure was recorded, including an
   evaluator disconnect, a failed finish or a client cancellation, the attempt
   still reports `integrity-check-failed`.

Cancellation, deadline, memory exhaustion, transport and admission failures
stop the route immediately, as before. A native parser or decryptor that
aborts the process rather than raising also stops it immediately.

### Tolerance

`tau(m) = ceil(2^15 * B(m) * z)`, with `z = sqrt(2 ln 2 * 41) = 7.54` and
`B(m) = margin * sqrt(80 + 7.5 m) * (sqrt(m) + z)`, `margin = 2`.

`B(m)` is the assumed bound on the Euclidean norm of an honest tile row's decode
error (assumption A1). The fit and the margin come from the
[measured sweep](evidence/integrity/honest-noise-ntt12.json): per-coordinate
variance is about `7.4 m` on uniform inputs, and one extreme-input case showed a
data-dependent bias that raised the largest row norm to 8,791 at `m = 2048`,
against `B(2048) = 13,120`. The installed policy must keep at least 16 vectors,
a sample numerator of at least one, a denominator of at most 64 and a margin of
at most 4. The policy digest is reported with each run; it is not part of the
plan identity.

## Claims

Write `delta = y - xW` for the deviation of a decoded row and `R = 2^15`.

**Claim 1 (linear check soundness, per tile row).** For any response chosen
without knowledge of the check secret, if some coordinate of `delta` has
magnitude `D > 0`, all `t` checks accept with probability at most
`((2 tau / D + 1) / 2R)^t`. Different rows and steps are independent trials.

*Argument.* `<r_k, y> - <x, u_k> = <r_k, delta>` exactly, because `u_k = W r_k`
and the arithmetic does not overflow. Let `j` be a coordinate with
`|delta_j| = D`. Condition on every other entry of `r_k`. Then
`<r_k, delta> = a + r_kj delta_j` for a fixed `a`, and `r_kj` is uniform over
`2R` consecutive integers, so the value ranges over `2R` points spaced `D`
apart. At most `floor(2 tau / D) + 1` of them lie in `[-tau, tau]`. The `t`
vectors are independent. The response is independent of the secret because
nothing derived from the secret is ever sent and, within an attempt, no accept
or reject outcome changes anything the evaluator can see (Claim 4). ∎

The reviewer's simulation found the bound tight: a single coordinate at
`1.5 tau / R` was accepted about 0.25% of the time.

Deviation that every check accepts with probability at most `2^-40`, and at
most `2^-16`:

| Output width `m` | `tau / R` | `D` for `2^-40` | `D` for `2^-16` |
| --- | --- | --- | --- |
| 128 | 9,167 | 51,863 | 18,336 |
| 640 | 34,588 | 195,678 | 69,179 |
| 896 | 46,592 | 263,590 | 93,188 |
| 2048 | 98,914 | 559,590 | 197,834 |

**What the check accepts with certainty.** Any deviation whose absolute values
sum to at most `tau / R` passes every check, because `|<r, delta>| <= R ||delta||_1`.
At `m = 2048` that budget is 17.8 times the measured honest row norm; at
`m = 128` it is about 26 times. In the reviewer's simulation with honest noise
added, at `m = 2048`: one coordinate shifted by `0.95 tau / R` (93,968 units,
about 760 honest standard deviations) was accepted 98% of the time; every
coordinate shifted by ten standard deviations in a chosen sign pattern, 98%;
by twelve, 87%. The table is per tile. A projection whose outputs sum `k` input
tiles (`down_proj` sums three) has `k` times the budget per output.

For scale, on uniform twelve-bit data the exact products have standard deviation
about `1.25e8`, so `tau / R` at `m = 2048` is about 0.08% of that and the `2^-40`
threshold about 0.45%. On the small-Qwen fixture the sampled tiles had root mean
square products between `1.1e8` and `9.8e8`, so the relative figures there are
similar or smaller.

**Claim 2 (honest responses are not rejected).** If an honest tile row's decode
error satisfies `||e|| <= B(m)` (A1), a single check rejects with probability at
most `1.004 * 2^-40`. A small-Qwen attempt runs 68,112 checks, so an honest
attempt is rejected with probability at most about `2^-24`.

*Argument.* For fixed `e`, `<r_k, e>` is a sum of independent terms bounded by
`R |e_j|`; Hoeffding gives `2 exp(-tau^2 / (2 R^2 ||e||^2))`. The entries of `r`
have mean `-1/2`, not zero, which costs the factor 1.004 at `m = 2048`. ∎

A1 is an empirical assumption about this native build and parameter profile,
not a theorem. If it fails, honest attempts are rejected; soundness is
unaffected. On real small-Qwen activations for the public capital fixture, 129
sampled tiles over the prefill and one decode step had row error norms of at
most 43% of `B(m)` ([calibration](evidence/integrity/qwen-ntt12-check-calibration.json)).

**Claim 3 (sampled replay).** If the evaluator returns anything other than the
exact honest bytes for `k` tile-steps, at least one is recorded as a failure
with probability `1 - (15/16)^k`, whatever the size of the change, provided
HMAC-SHA256 is a pseudorandom function (A3).

*Argument.* Each tile-step is sampled independently by a keyed function the
evaluator cannot evaluate. The client reads the complete response before it
starts any replay, so its read behaviour is the same for sampled and unsampled
tiles and the evaluator has committed to every byte before the client's replay
delays the next frame. Replay compares against the deterministic honest
ciphertext, so the outcome does not depend on the plaintext or on the size of
the change. An evaluator that infers from the delay that it was sampled cannot
change the result: a recorded failure outranks any way the route then ends. ∎

One tampered tile-step survives with probability 15/16. Sampled replay deters
sustained tampering; it does not prevent a single targeted change inside the
accepted region.

**Claim 4 (what an attempt reveals through integrity).** Within an attempt the
evaluator's view is the same whether or not any integrity check failed, up to
the encryptions themselves (A2) and client timing. Per attempt it can learn the
answer to one yes/no question of its choosing about the private input, and only
if it can observe whether the attempt was accepted.

*Argument.* Failures are recorded locally. The client sends the same number of
frames with the same kinds, identities and lengths, then the same finish. The
focused test compares the complete wire trace of an honest and a tampered
attempt. Plaintexts differ after a zero substitution, which A2 hides. ∎

The question is sharp and is answered by acceptance as much as by rejection. An
evaluator can return a response that is correct only when a linear function of
`x` is near zero. In the reviewer's experiment a deviation of `46 x_i` on one
output at `m = 896` was accepted always at `x_i = 900`, 27% of the time at 1,100
and never at 1,600. Over `n` attempts on inputs that share private content this
is `n` such answers.

There is a second rejection that the evaluator can influence on this route. The
numerical reference check after close (`token-reference-mismatch`) can be made
to fail by a deviation inside the accepted region. On the replay route no
evaluator-chosen deviation reaches that check.

**Operating rule.** Any rejection after close on a checked route, with either
code, ends remote evaluation of that input and of anything that shares private
content with it, for every remote operator, since operators may collude. The
research driver has no automatic retry. This rule is not enforced in code.

With immediate abort the position of the first failing tile would also be
visible, about 17 bits per attempt on the Flash route. Deferral removes that.

**Claim 5 (rejected content does not feed forward).** After a rejected
response, no later client plaintext depends on that response's content.

*Argument.* The tile's contribution is replaced by zeros before accumulation.
The unparsed or decrypted value is dropped. The focused test checks that the
result equals the honest result minus exactly that tile's contribution. ∎

## What this gives up relative to full replay

| Property | Full replay | Checked route |
| --- | --- | --- |
| Accepted response | Exact honest bytes | Decodes inside the accepted region around `xW`; exact only on the sample |
| Client parses unverified bytes | Never | Yes, after the length gate |
| Client decrypts unverified ciphertexts | Never | Yes |
| Outcome depends on private data | No | Yes; deferred, one chosen yes/no answer per attempt |
| Evaluator can cause a numerical-reference rejection | No | Yes |
| Client encrypted work | Every tile | First of each geometry plus 1/16 of tiles |

Consequences that are stated, not solved:

- **Steering inside the accepted region.** The region is many times honest
  noise and the evaluator chooses the direction. Whether a model's output can
  be steered this way, and how many tiles it takes, is a property of the model
  and is not established. Sampled replay is the only defence, and it is
  probabilistic.
- **Key-dependent plaintexts (A4).** Valid ciphertext operations can add a small
  multiple of a key bit to an output and stay inside the accepted region, so a
  later client plaintext can depend on the secret key. The evaluator sees only
  encryptions of it. This relies on the encryption remaining secure for such
  messages, an assumption the replay route did not need.
- **Parser exposure.** The native deserializer panics on malformed input and
  trusts size fields inside a correct-length response; one corrupted field made
  it expect a 201 MB buffer before failing an assertion. Panics that surface as
  exceptions are contained and recorded as rejections. No memory-safety or
  resource-bound claim is made for the parser.
- **No repeated decryption queries against one key.** Keys are fresh per
  attempt. This is an argument, not a proof of security against chosen
  ciphertexts.
- **Deployment evidence.** Checked routes have their own SSH route schema and
  frame protocol. No checked route has been run across machines; every retained
  measurement is same-host.
- **Flash.** The Flash checked profile has a plan loader, admission tests and
  bridge admission. No driver executes it yet.

## Cost

The client still does, per attempt: one exact product `W r` per tile row (16
integer vectors), one decryption and 16 inner products per tile row, and an
encrypted replay of the first response of each geometry plus a 1/16 sample. It
no longer replays the remaining tiles.

The check needs the client to hold `W`. A client that holds `x` and `W` could
also compute `xW` itself with no evaluator. The checked route makes the client's
integrity work much cheaper than replay; it does not make the hybrid route
cheaper than local clear execution. One complete small-Qwen generation through
the checked route is [retained](FULL-QWEN.md#checked-integrity-route): client
encrypted replay fell from 89.71 to 5.89 minutes against the replay receipt,
with 14.8 seconds of linear checks.

## Tests

| Concern | Test |
| --- | --- |
| Soundness bound, completeness, per-row independence, certain-acceptance budget, exact arithmetic, domain guards including the int64 extremes, policy floors | `test_linear_check.py` |
| Deferred rejection with an unchanged wire trace, exact zero substitution, no replay while a response is pending, failure precedence over evaluator-chosen endings, cancellation, public-weight change, earlier-step replay, secret never on the wire, replay-plan refusal | `test_checked_bridge.py` |
| Checked SSH route schema and protocol cannot be exchanged with the replay ones | `test_ssh_route.py` |
| Real ciphertexts: wrong matrix, wrong input, wrong key, wrong length, bit flips, the accepted-region gap, sampled replay | `test_checked_native.py` (opt-in) |

The native bit-flip test flips 464 distinct bits of a real response. Every
accepted flip decoded within the Claim 1 threshold; the rest were rejected.

## Assumptions

- **A1.** Honest decode error of a tile row has Euclidean norm at most `B(m)`.
  Measured on synthetic tiles and, with `calibrate_linear_check.py`, on one
  small-Qwen fixture. Not measured on Flash.
- **A2.** The GLWE encryption hides plaintexts from the evaluator under the
  installed parameters. Unreviewed; see the security boundary.
- **A3.** HMAC-SHA256 and SHAKE-256 keyed with the check secret are
  pseudorandom, and the operating system's random source is unpredictable.
- **A4.** Encryption stays secure when later plaintexts depend on the key
  through an accepted response.

## Open before any admission

Review by a cryptographer of this argument and of A2 and A4; a measured A1 on
real model activations for each admitted geometry; a model-level bound or
measurement for steering inside the accepted region; parser hardening or a
verified length-and-structure gate; the operating rule enforced in code;
timing analysis of the deferred path; a checked run across machines; and
everything already listed in the security boundary.

## Review record

**2026-11-17, independent automated review.** A separate Claude session with no
access to the author's reasoning reviewed the document, the check arithmetic,
the bridge, the driver and the tests at commit `88e2a6268`, and ran its own
numerical experiments. It is not a cryptographer's review and does not qualify
the mechanism.

It confirmed the Claim 1 bound and its tightness, the tolerance formula, exact
integer arithmetic under the guards, unbiased sampling, wire invariance under
failure, and that no rejected plaintext is used. It could not check native
behaviour, A1 on real activations, A2 or A4.

| Finding | Disposition |
| --- | --- |
| The client replayed while the response was pending, so a slow read revealed the sample before the evaluator had sent the whole response; tampering only unsampled tiles went undetected in 15 of 15 trials for responses above the pipe buffer | Fixed: replay starts only after the complete response is in memory |
| A caught evaluator could end the route so the attempt reported a transport error instead of an integrity failure, and then be retried | Fixed: a recorded failure outranks every other ending |
| "At most one bit" and the retry rule were too narrow: acceptance leaks too, the numerical reference check is a second evaluator-influenced rejection, and operators may collude | Claim 4 and the operating rule restated |
| "Noise-sized steering" understated the accepted region; vectors shared across rows and steps let one pattern persist at one check's risk | Region stated with measured ratios; vectors are now independent per tile, step and row |
| The default suite did not test zero substitution | Test now asserts the exact result on an unsampled tile |
| The protocol guard rejected every SSH-bound plan, and SSH frames named replay | Checked routes have their own SSH route schema and protocol |
| The domain guard missed the most negative int64 | Fixed: extremes compared directly |
| Claim 2 constant and per-attempt false-reject rate unstated | Stated |
| The linear-check policy could be weakened without changing any identity | Floors enforced; the digest is reported, not bound into the plan |
| Not every parse or decrypt failure is deferred | Stated |
| The cost link pointed at a missing section; one profile was named, two exist | Corrected |
