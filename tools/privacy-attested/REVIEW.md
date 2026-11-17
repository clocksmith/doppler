# Independent review, 2026-11-17 (findings not applied)

This branch is parked. The owner ruled out hardware attestation as the direction for the protected tier the same day
it was written; the tier continues as two-server secret sharing on the mesh. Nothing here is on `main`. The findings
below came from an independent automated review of this code and **have not been fixed**. Fix them before reusing any
of it.

| # | Severity | Finding | Fix |
| --- | --- | --- | --- |
| 1 | High | `snp-endorsement.js`: the chip certificate is verified by `X509Certificate`, which accepts PEM and skips leading bytes, while the hardware ID and firmware versions are read by walking the raw input from offset 0. A forged DER prefix followed by a genuine certificate in PEM was accepted with attacker-chosen hardware ID and firmware versions. With any one leaked or old chip key this defeats the chip allowlist and firmware minimums. Demonstrated on the endorsement step; not end to end. | Require `vcek.raw` to equal the input and walk `vcek.raw`. |
| 2 | Medium | Launch firmware version is parsed but never compared with the deployment minimum; platform-info bits are unused. | Require launch TCB at or above the minimum; decide the platform bits per deployment. |
| 3 | Medium | `client.js` `verifyEvidence` does not await the verifier, so an asynchronous verifier that rejects is treated as success. | Await the result and require a frozen appraisal of the expected shape. |
| 4 | Medium | Many appraisal, session and lifecycle checks can be deleted without any test failing; three test names claim more than they assert (partial text is delivered before a tampered frame is caught; cancellation is triggered from a delivered delta). | Add a mutation-style test per check; rename or tighten the three tests and the matching lines in DESIGN.md. |
| 5 | Low to medium | The host chooses the evidence format, and any registered verifier is trusted for sealing. | Refuse `sealInput` unless isolation is established, with an explicit test-only override. |
| 6 | Low | A host can make the client or the guest buffer unbounded data (frame size is checked late; queued lines are unbounded). | Check frame size on receipt; cap or pause the queue. |
| 7 | Low | Malformed hello, a failing report request, or a malformed line make `serve` reject without closing the channel. | Type-check the namespace; wrap the whole body so every exit refuses and closes. |
| 8 | Low | Only derived keys are zeroed; the X25519 private keys and plaintext copies are not. | Destroy the key objects and buffers, or drop the zeroization from the description. |
| 9 | Low | A response of exactly the frame limit ends without its authenticated status. | Reserve one frame for the final status. |
| 10 | Low | Trust-policy validation cannot tell an empty chip list from an intended "any chip". | Make "any" explicit. |
| 11 | Low | Report versions above 3 are refused (availability only). | Add them when a platform needs them. |

Judged sound by the same review: report layout and signer handling, the root pins and chain check, the report-data
binding, the key schedule and frame authentication, error text on the wire, and the installed policies admitting
nothing. Not checked: anything on real hardware, revocation, Turin, fuzzing of the certificate walker.
