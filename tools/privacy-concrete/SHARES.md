# Two-server shares route: design and threat model

Status: research. Production admission remains closed. No cryptographer has reviewed this. The
[review record](#review-record) lists what has been checked.

## Decision

Owner decision, 2026-11-17: the protected tier must leave the requester light and must run on the mesh's ordinary
computers, with no special chip features. The encrypted Flash run on the homomorphic route in this directory is
stopped; its small-Qwen receipts are kept unchanged as research.

## Idea

Almost all of a transformer's weights and arithmetic sit in its linear layers. Everything else (embedding lookup,
normalisation, attention mixing, activation functions, sampling) is cheap and needs no large matrix.

For each linear layer the requester holds a row `x` of twelve-bit integers. All arithmetic is modulo the prime
`P = 2^61 - 1`. The requester draws `x1` uniformly at random and sets `x2 = x - x1`. Provider A receives `x1`, provider
B receives `x2`. Each multiplies its row by the public integer matrix `W`. The requester adds the two answers:
`x1 W + x2 W = x W`, exactly. Each provider only ever sees uniformly random numbers.

A prime modulus is used instead of 2^64 because a prime has no zero divisors: a provider cannot make a deviation
vanish by multiplying it by a power of two.

| | Homomorphic route | Two-server shares |
| --- | --- | --- |
| What hides the input | GLWE encryption | Splitting it between two providers who do not collude |
| Requester holds linear weights | Yes | No |
| Requester cryptography | Encrypt, decrypt and re-check every tile | Random numbers, additions and a few modular multiplies per row |
| Provider arithmetic | Homomorphic, on CPU | Ordinary matrix product, reduced modulo the prime |
| Result | Approximate, decode noise | Exact |
| Keys | 268 MB evaluation key | None |
| Special hardware | None | None |

The approach follows Gupta, Katz and Miers, *Private and Verifiable Outsourcing of Open-Weight LLM Inference*
(IACR ePrint 2026/1849), which delegates the linear layers of an open-weight model to two non-colluding servers and
leaves the non-linear layers with the client. The check below is this project's own and has not been compared with
theirs in detail; only their abstract and secondary summaries were read.

## Parties

- **Requester.** Holds the prompt and the output. Runs the client model with the `shared` mode of
  [full_model.py](full_model.py) and [shares.py](shares.py). Trusted.
- **Providers A and B.** Two mesh computers, each running [shares_evaluator.py](shares_evaluator.py) with the public
  coefficient files. Each may be dishonest. They must not pool what they see.
- **Model owner.** Publishes the weights and the manifest.

The requester keeps a secret 32-byte seed. From it, the projection's name and its width, each projection gets a
secret probe row `p`, and the requester
holds that projection's anchor `p W`: one number per output column. Computing the anchors takes one pass over the
public weights by the requester or by someone it trusts. The seed may be kept for as long as the anchors are; the
retained runs drew a fresh one per run and computed the anchors in the same process, standing in for that pass.

## Protocol for one projection

With `R` input rows and `c` check rounds (installed: 1):

1. The requester draws uniform masks `t_k` and `u_k` (one `R`-row block per round) and, per row and round, secret
   multipliers `alpha_k`, `beta_k` and non-zero multipliers `delta_k`, `epsilon_k`.
2. A receives `x1`, then `alpha_k x1 + delta_k p + t_k`, then `u_k`.
   B receives `x2`, then `beta_k x2 + epsilon_k p + u_k`, then `t_k`.
3. Each provider returns its rows times `W` modulo `P`.
4. The requester accepts only if, for every round, A's check block minus `alpha_k` times A's real block minus B's
   answer on `t_k` equals `delta_k` times the anchor, and the mirror holds for B. It also checks that the sum of the
   two real blocks lies inside the range an honest product can reach; that is a sanity bound, not the mechanism.
5. A failed check is recorded locally. The requester substitutes zeros and carries on, closes both connections at the
   end and only then reports `integrity-check-failed`, releasing nothing.

Blocks can be told apart by their position in a message, but every row is masked by fresh uniform randomness, and the
probe never travels alone.

## Claims

Assumptions: **S1** the two providers do not share their views or coordinate. **S2** the requester's random numbers
and probe seed are unpredictable and stay secret. **S3** the requester's anchors were computed from the true weights.

- **Claim 1, privacy.** Each provider's rows are jointly uniform over the field and independent of the input, the
  probe and the multipliers, so one provider alone learns nothing about the activations, whatever it computes. This
  needs no hardness assumption; field elements are drawn exactly uniformly, redrawing the one out-of-range value. Argument: `x1` is uniform; `x2 = x - x1` is
  uniform because `x1` is; every other row carries its own fresh uniform mask. [test_shares.py](test_shares.py) shows
  that A's rows are identical for a different input, that B's rows are identical for a different input under an
  equally likely first share, and that A's rows are identical for a different probe under equally likely masks.
- **Claim 2, one dishonest provider is caught.** If one provider changes any value in its real block while the other
  is honest, the attempt is rejected except with probability about 2^-61 per altered row and round (reported
  conservatively as `singleEvaluatorDetectionLog2`, 60). Argument: to pass, its change to the check block must equal
  `alpha_k` times its change to the real block in the field, and it has never seen `alpha_k`.
- **Claim 3, the same deviation by both providers is caught.** If both providers return zeros, skip blocks, or
  multiply by the same other matrix `W + D`, each check is off by `delta_k` times `p D`. In a field that is zero only
  if `p D` is zero, which for a `D` chosen without knowing the probe happens with probability about 2^-61 per column
  (`commonDeviationDetectionLog2`, 60). More rounds do not improve this figure, because every round uses the same
  probe. This covers providers that deviate in the same way without communicating, for example both skipping the
  work. Providers that pool their views are outside S1.
- **Claim 4, a caught provider learns nothing from being caught.** Rejection is deferred, and whether a check fails
  depends on the providers' own changes and the requester's secrets, not on the input. Two exceptions. The sanity
  bound depends on the input, but a wrong value reaches it only after passing the field checks, with the
  probabilities above. And after a failure the requester computes on substituted zeros; if a later client step then
  cannot proceed, the route closes early, and the number of calls made before that could depend on the input. That
  second exception is not measured.

## What is not defended

- **Collusion.** If A and B combine their rows they recover every activation, and from those the prompt. Choosing two
  providers under different operators is a placement decision for Reploid and an assumption here, not something the
  protocol can verify. Separate signing keys do not prove separate operators.
- **Shape and timing.** Providers see the projection order, the number of rows (the prompt-length bucket), message
  sizes and timing.
- **Model confidentiality.** Weights are public by design.
- **Availability.** Either provider can stop the attempt. A provider that stops reading or answering is cut off at a
  per-message deadline that ends its whole process group, closing is bounded, and an attempt has a one-hour limit. Each in-flight send or receive uses the remaining absolute attempt budget,
  capped by the per-message timeout; cancellation also ends blocked provider I/O. Failed attempts kill both provider
  process groups before bounded cleanup.
  None of that makes a slow provider fast.
- **Transport and identity.** This prototype talks to providers over process pipes or SSH. Provider identity, grants
  and receipts belong to Reploid and are not built.
- **Flash.** Evaluating only the routed experts would show the providers which experts a token chose. Flash's large
  lookup tables would need a two-server private lookup. Neither is designed.
- **What the requester still stores.** The embedding table (545 MB at 32-bit for small Qwen), normalisation
  parameters, and per projection a few weight-derived vectors (quantization scales, zero points, column sums, bias)
  and the anchor row. These come from the model owner and are trusted like the weights. It stores no weight matrix.
- **GPU tuning.** The GPU evaluator (below) is exact and reuses its workspace buffers, uniform slots and bind groups
  across calls, but each linear layer still executes as a synchronous GPU submit and readback fence (`169` calls per
  decode step), which dominates on small matrices.

## GPU evaluator

[shares_evaluator_gpu.js](shares_evaluator_gpu.js) is a drop-in provider that speaks the same pipe protocol as the
CPU one and runs the product in [shares_field_matmul.wgsl](shares_field_matmul.wgsl) through a Node WebGPU binding.
WGSL has no 64-bit integers, so each field element is uploaded as four 16-bit limbs and each weight as an unsigned
16-bit value (the signed weight plus 32768). A limb times a weight fits exactly in 32 bits; the sum over the input
width carries into a second word; the four limb sums are rotated into place modulo the prime (multiplying by 2^16k
is a 61-bit rotation), added, and corrected for the weight offset. Workspace storage, staging and uniform buffers grow
to the process high-water mark and are reused across calls. The answer is the same eight bytes per value the CPU
provider returns.

[test_shares_gpu.py](test_shares_gpu.py) compares it with the CPU reference bit for bit on four shapes, including an
input width above 4096, weights at both 16-bit and 8-bit extremes, multi-chunk column dispatches, workspace reuse
across growing and shrinking batches, symlink and invalid-geometry rejection, mid-frame stream truncation, and inputs
of zero, the prime minus one, the prime and 2^64 - 1, and runs a complete checked route on two GPU providers and on one
GPU provider beside one CPU provider. It needs a WebGPU binding and is skipped without one. Breaking one rotation in
the kernel makes two of its tests fail.

## Measured

Two retained runs of the pinned small-Qwen fixture (four output tokens, eight prefill rows, 169 linear layers per
step), with their [context](evidence/shares/qwen-shares-context.json):

| | [One host](evidence/shares/qwen-shares-local.json) | [Three machines](evidence/shares/qwen-shares-three-machine.json) |
| --- | --- | --- |
| Placement | Requester and both providers on one Linux host | Requester on the Linux host, provider A on a Mac, provider B on a second Linux machine |
| Output tokens | 12095, 13, 1084, 374 (" Paris. It is") | the same |
| Largest logit difference from the quantized reference | 0 at every step | 0 at every step |
| Execution | 42.4 s | 163.9 s |
| Waiting for providers | 26.7 s | 153.5 s |
| Requester share arithmetic | 5.6 s | 3.1 s |
| Requester non-linear steps and orchestration | 10.1 s | 7.4 s |
| Sent to each provider | 65.2 MB | 65.2 MB |
| Received from each provider | 120.4 MB | 120.4 MB |

The tokens and the plan and model identities are the same as in the retained homomorphic receipts. The
[checked homomorphic run](evidence/integrity/qwen-ntt12-check-local.json) took 89.1 minutes for the same fixture, with
a 268 MB evaluation key; that is a different mechanism with a different guarantee, so the two are shown side by side
and not as a matched benchmark.

The three-machine run is the first with the requester and each provider on different computers. Its time is
dominated by moving about 370 MB through two SSH pipes and by each provider hashing its coefficient files on first
use; it is not a tuned figure. The requester stored 544.5 MB of embedding table, 5.6 MB of per-projection
quantization vectors and biases, 3.6 MB of anchors and 0.2 MB of normalisation parameters. Its one-time anchor pass
over the public weights took 1.8 s. The providers held the 169 weight matrices (1.98 GB at 32-bit).

**With GPU providers.** On a second, faster Linux host with an integrated AMD GPU, the complete fixture with the
requester and two GPU provider processes on that one host gave the same four tokens with zero difference in 7.7 s
before buffer reuse ([receipt](evidence/shares/qwen-shares-gpu-one-host.json)) and 7.5 s with persistent workspace
buffers ([receipt](evidence/shares/qwen-shares-gpu-reused-buffers.json)): 4.5 s for the first step, which includes each
provider verifying, transposing and uploading all 169 matrices, then 0.98 to 1.00 s per decode step (down from 1.03 to
1.05 s before buffer reuse). The same host with two CPU provider processes took 5.3 s and about 0.90 s per decode step
([receipt](evidence/shares/qwen-shares-cpu-fast-host.json)). On single products of that model the GPU provider is
faster on repeat calls: 0.060 s against 0.376 s for three rows through the 896 by 151,936 vocabulary head, 0.017 s
against 0.104 s for 24 rows through a 4,864 by 896 projection, and 0.003 s against 0.020 s for 24 rows through an
896 by 896 projection. It is not yet faster end to end on this small model because a decode step makes 169 synchronous
GPU submit and readback round-trips across two processes sharing one GPU.

On one host each decode step took 6.4 to 7.0 s on a two-core laptop processor shared by the requester and both
providers. All runs had one operator and no misbehaving provider, so they show that the route is exact and what it
costs, not that it is private in deployment or that it detects cheating; the tests cover detection. Earlier versions'
receipts are in this repository's history at commits `9975f02c5` and `8b73b5991`.

## Review record

- 2026-11-17, automated tests: 18 CPU pass ([test_shares.py](test_shares.py)) and 5 GPU pass
  ([test_shares_gpu.py](test_shares_gpu.py), run on both Metal and Vulkan WebGPU bindings).
- 2026-11-17, first independent automated review, of a version with a public probe in a separate row and arithmetic
  modulo 2^64. It demonstrated that two providers who both returned zeros, or both used the same other matrix except
  on the recognisable anchor row, were accepted 300 times out of 300, and that the range check accepted the most
  negative 64-bit value. Fixed by making the probe secret and hiding it inside every check row.
- 2026-11-17, second independent automated review, of that fix. It found no accepted wrong result. It demonstrated
  that with a power-of-two modulus two providers using the same matrix with one entry changed by 2^63 made acceptance
  depend on the parity of an input value for half of all probes, and that the detection figure for a common
  deviation was about 2^-29 at best and did not improve with rounds. It also showed that the deadline did not cover a
  provider that stops reading, and listed mutations the tests did not notice. This version answers those: arithmetic
  is in a prime field, the deadline covers sending, and the tests cover the demonstrated strategies and the listed
  mutations for seed dependence, mask structure, multiplier width, and a caught provider that then leaves.
- 2026-11-17, third independent review, by a different model family (an Antigravity session), of the prime-field
  version. It attacked Claims 2 and 3 in the field and found them sound under S1 to S3, and confirmed the float64 limb
  product and the modular helpers exact at the admitted bounds. It demonstrated six robustness defects, none letting a
  wrong result through: closing could hang on a provider that filled the pipe and slept; the deadline killed only the
  direct child, so a helper holding the pipes defeated it; the evaluator's row limit did not follow an overridden
  policy; the probe derivation did not bind the width or separate seed from name, and the field draw had a small
  bias; the route did not bind input widths or bound an attempt; and four tests asserted less than their names said.
  All are fixed here: providers run in their own process group and closing is bounded, the route always uses the
  installed policy and has an attempt limit, the probe binds seed, width and name, draws are exactly uniform, input
  widths are checked, and the tests were tightened and extended.
- 2026-11-18, independent re-review of the third-review fixes (`0595f8415`) and the in-flight deadline and cancellation
  fix (`d39b135ba`). Verified that watchdog expiry and normal completion serialize under one lock without racing later
  exchanges, that blocked sends, blocked reads and helper processes holding inherited pipes are terminated by process
  group within the bounded deadline, that a previously recorded integrity failure cannot be masked by a subsequent
  disconnect or timeout, that polite close uses an independent five-second grace period, and that probe derivation and
  rejection-sampled field draws match their contracts.
- 2026-11-18, independent review of [shares_field_matmul.wgsl](shares_field_matmul.wgsl) and
  [shares_evaluator_gpu.js](shares_evaluator_gpu.js). Verified the four-limb unsigned-offset field product, 61-bit
  rotations (`rotate16`, `rotate32`, `rotate48`), `reduce61` fold bound (`< 2^63`), and borrow-free Mersenne offset
  negation exact for all admitted widths (`inputs < 2^21`). Found and fixed (`7566d2142`) missing manifest geometry and
  32-bit index bounds, `maxComputeWorkgroupsPerDimension` column-chunk capping, odd-offset `Int16Array` buffer
  alignment, `int8` (`|i1`) weight support, per-call GPU buffer and per-element `BigInt` allocations, and uncaught
  `end of stream` rejections when stdin closes mid-frame.
- Cryptographer review: none.
