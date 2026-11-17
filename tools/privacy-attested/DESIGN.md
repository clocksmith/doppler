# Attested light-client route: design and threat model

Status: parked research branch, not on `main`. The owner ruled out hardware attestation as the direction on 2026-11-17, and an independent review found defects that are not fixed ([REVIEW.md](REVIEW.md)). Production admission remains closed. The protocol has run only against a software stand-in for the
hardware; no confidential guest has been provisioned and no cryptographer has reviewed it. The
[review record](#review-record) lists what has been checked and by whom.

## Decision

Owner decision, 2026-11-17: the protected execution tier is re-scoped around a light client. The requester must not do
model work. The encrypted Flash run on the [hybrid GLWE route](../privacy-concrete/README.md) is stopped; its
small-Qwen receipts stay as research evidence and are not changed.

The hybrid route keeps the prompt hidden by having the client run every non-linear model step and hold the weights,
while a remote evaluator multiplies encrypted vectors by public matrices. That makes the client the expensive party by
construction. This route moves all model work back to the provider and protects it with hardware isolation instead.

| | Hybrid GLWE route | Attested route |
| --- | --- | --- |
| Client holds weights | Yes, all of them | No |
| Client model operations | Attention, normalisation, sampling and the integrity check | None |
| Client work per attempt | 5.89 min of sampled replay on small Qwen, four tokens ([receipt](../privacy-concrete/evidence/integrity/qwen-ntt12-check-local.json)); 89.71 min with full replay | About 14 ms of CPU for a 257-frame answer ([local run](evidence/local-software-run.json)) |
| Provider speed | Homomorphic arithmetic on CPU | The ordinary plaintext runtime |
| What the requester must trust | The GLWE parameters and its own machine | The CPU vendor's isolation and signing keys, and the measured guest software |
| Model confidentiality | None, weights are public | Possible, weights stay with the provider |

The two rows of client work are not a matched benchmark: they are different models, hosts and guarantees. They show
the size of the difference, which is the reason for the decision.

## Parties

- **Requester.** Holds the prompt and wants the answer. Runs [client.js](client.js). Trusted.
- **Guest.** A confidential virtual machine that runs the Doppler runtime and the model. Runs [provider.js](provider.js).
  Trusted only after its hardware evidence has been appraised.
- **Host.** The provider's operator, hypervisor, other tenants and the network. Sees and can rewrite, drop, reorder and
  replay every message. Untrusted. All remote operators may collude.
- **Hardware vendor.** AMD signs each chip's key. Trusted; this is the assumption the tier rests on.

## Protocol

`doppler-attested-session/v1`, one generation per session.

1. **Hello.** The requester sends a fresh 32-byte nonce, the model identity, the
   [state namespace](../../src/inference/execution-backend-contract.js) and the session policy digest. All public.
2. **Evidence.** The guest makes a fresh X25519 key and asks the hardware for a report over
   `SHA-512(tag ‖ nonce ‖ guest key ‖ model ‖ namespace ‖ session policy)`. It returns the report, its chip
   certificate and the key.
3. **Appraisal.** The requester checks, in order: the chip certificate chains to a pinned AMD root; the report is
   signed by that chip key; the report's chip and firmware versions match the certificate; the guest forbids debugging
   and migration agents; the launch measurement, chip and firmware minimums match an admitted deployment; and the
   report's 64 data bytes equal the value in step 2, recomputed from the requester's own nonce and expectations.
4. **Seal.** Only after step 3 succeeds the requester makes its own X25519 key, derives one key per direction with
   HKDF-SHA256 (salted by a transcript of the binding, the report and its key) and encrypts the request with
   AES-256-GCM.
5. **Execute.** The guest decrypts, runs ordinary plaintext generation and streams encrypted frames. Each frame
   authenticates its position, direction and whether it is the last.
6. **Finalize.** The requester releases the text when the authenticated final frame arrives.

## What is bound before disclosure

Step 3 fixes all of these before one request byte is encoded:

| Identity | How it is bound |
| --- | --- |
| Model, and through its manifest every tensor | Model digest inside the signed report data |
| Guest software | Launch measurement, matched against the trust policy |
| Recipient | Chip ID and host data, matched against the deployment when it lists them |
| Key | Guest X25519 key inside the signed report data |
| Attempt | Requester nonce inside the signed report data; fresh keys on both sides |
| Tenant, session, policy, route, adapters | State namespace digest inside the signed report data |
| Framing and padding rules | Session policy digest inside the signed report data |

## Threat model

The host can do everything listed under Parties. It cannot break SEV-SNP memory isolation, forge AMD signatures or
break X25519, HKDF or AES-GCM (assumption A1 and A2 below).

Defended, each with a test in [attested.test.js](attested.test.js):

- **Impersonation.** A host that answers with its own key, a different model, or a genuine report recorded from an
  earlier session fails the binding check, and no request is sent.
- **Wrong guest.** A debuggable guest, a different measurement, old firmware or an unlisted chip fails appraisal.
- **Tampering.** An altered request is refused by the guest. An altered, dropped, repeated or re-flagged response
  frame fails authentication and the text is not released.
- **Replay.** A recorded request does not decrypt in another session, because both keys are fresh.
- **Truncation.** A stream without its authenticated final frame is an error.
- **Cancellation.** Aborting stops the guest's generation and releases nothing.
- **Leakage through errors.** Error codes on the wire are fixed strings. A failure inside the guest is reported as a
  status, never as text.

Not defended, and stated so:

- **The hardware assumption itself.** Published attacks on confidential VMs exist and are patched through firmware;
  the firmware minimums in a deployment entry are the only control here. A break of SEV-SNP breaks this tier.
- **Side channels.** Message count and timing reveal the number of output pieces and how long each took. Request
  length is hidden only up to a 4,096-byte bucket and each response frame up to 256 bytes.
- **Certificate revocation.** AMD's revocation list is not fetched or checked.
- **Guest software defects.** The measurement says which software runs, not that it is correct. What the guest logs or
  stores is governed by that software.
- **Accelerators.** A GPU the guest uses must itself be inside the protected boundary. A guest that sends activations
  to an ordinary GPU has left the boundary, and nothing in a CPU report reveals that.
- **Availability.** The host can always refuse service.
- **Requester authentication.** Any party may start a session. Authorising requesters is Reploid's grant layer.

## Claims

Assumptions: **A1** SEV-SNP isolates guest memory from the host and only genuine AMD firmware can produce a report
that verifies under a chip key AMD endorsed. **A2** X25519, HKDF-SHA256, AES-256-GCM and SHA-512 have their standard
properties. **A3** The measured guest software keeps its session private key and plaintext inside the guest. **A4**
The requester's machine is not compromised.

- **Claim 1, the key belongs to the guest.** Under A1 and A2, if appraisal succeeds then the X25519 key in the evidence
  was placed in report data by software with the admitted measurement, in response to this requester's nonce. Argument:
  the report signature covers the report data; the report data is a collision-resistant hash over fixed-length fields
  that include the nonce and the key; the nonce is 32 fresh random bytes, so the report was produced after the hello.
- **Claim 2, confidentiality in transit.** Under A1 to A4, the host learns nothing about request or response text
  beyond the leakage listed above. Argument: both directions are encrypted under keys derived from an X25519 secret
  between the requester's fresh key and the key from Claim 1.
- **Claim 3, the released text is the guest's text.** Under A1 to A3, text returned by `finalizeClient` is exactly the
  sequence of pieces the guest sealed for this session, complete and in order. Argument: each frame's position,
  direction and final marker are authenticated data under a key only the two ends hold, and the transcript salt ties
  the keys to this session's report.
- **Claim 4, no silent downgrade.** The protocol has its own identifier and descriptor kind. Doppler's
  `assertExecutionBackendAvailable` still refuses every attested descriptor, and the installed trust policy lists no
  deployment, so the installed verifier admits nothing.

Claim 3 is about transport. Whether the guest ran the model correctly rests on the measurement, which is weaker than
the arithmetic check the hybrid route had and far cheaper.

## Client cost

[One local run](evidence/local-software-run.json), requester and stand-in provider as two processes on one Mac:
14.4 ms of client CPU in total for a 2,000-character request and a 257-frame response, of which evidence appraisal took
1.0 ms and opening the response 2.8 ms. The wire carried 5,965 bytes to the provider and 102,503 back. Checking a
genuine Milan report through AMD's chain takes a median 0.55 ms of CPU on the same machine. The stand-in ran no model,
so the run says nothing about provider speed, which is the ordinary runtime's.

## Hardware

No machine in the owner's fleet can host this tier today. SEV-SNP exists on AMD EPYC server processors, not on the
fleet's desktop and laptop chips, and an earlier inspection found no guest attestation device on the AMD host. A real
deployment therefore needs rented or purchased confidential hardware, and for a model of Flash's size a GPU that is
itself inside the protected boundary. Choosing that hardware is an owner decision and is the next gate.

## Before admission

1. Provision one confidential guest, record its launch measurement and firmware, and add it as a deployment.
2. Run [attester-configfs.js](attester-configfs.js) on it. It has never run on hardware.
3. Run a real model in the guest and measure provider speed, memory and the complete cost against a plaintext provider.
4. Decide the accelerator question for the target model.
5. Add revocation checking, and review the firmware minimums against AMD's security bulletins.
6. Decide padding and timing policy for the leakage listed above.
7. Independent cryptographic and implementation review.
8. Reploid integration: policy presets, grants and requester receipts for the attested kind.

## Review record

- 2026-11-17, automated tests: 17 pass, including two genuine AMD reports verified through AMD's chain.
- 2026-11-17, independent automated review: one high-severity and three medium findings, **not applied**; see [REVIEW.md](REVIEW.md). This branch is parked.
- Cryptographer review: none.
