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

For each linear layer the requester holds a row `x` of twelve-bit integers. It draws `x1` uniformly at random from the
integers modulo 2^64 and sets `x2 = x - x1`. Provider A receives `x1`, provider B receives `x2`. Each multiplies its
row by the public integer matrix `W` modulo 2^64, which is an ordinary matrix product. The requester adds the two
answers: `x1 W + x2 W = x W`, exactly. Each provider only ever sees uniformly random numbers.

| | Homomorphic route | Two-server shares |
| --- | --- | --- |
| What hides the input | GLWE encryption | Splitting it between two providers who do not collude |
| Requester holds linear weights | Yes | No |
| Requester cryptography | Encrypt, decrypt and re-check every tile | Random numbers, additions and one multiply per row |
| Provider arithmetic | Homomorphic, on CPU | Ordinary integer matrix product |
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

The requester keeps a secret 32-byte seed. From it, each projection gets a secret probe row `p`, and the requester
holds that projection's anchor `p W`: one number per output column. Computing the anchors takes one pass over the
public weights, once per model, by the requester or by someone it trusts. In the retained runs the same process
computed them, standing in for that one-time pass.

## Protocol for one projection

With `R` input rows and `c` check rounds (installed: 2):

1. The requester draws uniform masks `t_k` and `u_k` (one `R`-row block per round) and, per row and round, secret
   multipliers `alpha_k`, `beta_k` and odd multipliers `delta_k`, `epsilon_k`.
2. A receives `x1`, then `alpha_k x1 + delta_k p + t_k`, then `u_k`.
   B receives `x2`, then `beta_k x2 + epsilon_k p + u_k`, then `t_k`.
3. Each provider returns its rows times `W` modulo 2^64.
4. The requester accepts only if, for every round, A's check block minus `alpha_k` times A's real block minus B's
   answer on `t_k` equals `delta_k` times the anchor; the mirror holds for B; and the sum of the two real blocks lies
   strictly inside the range an honest product can reach.
5. A failed check is recorded locally. The requester substitutes zeros and carries on, closes both connections at the
   end and only then reports `integrity-check-failed`, releasing nothing. If a later client step cannot proceed, the
   route closes early with the same outcome.

No row is recognisable: every row a provider receives is masked by fresh uniform randomness, and the probe never
travels alone.

## Claims

Assumptions: **S1** the two providers do not share their views or coordinate. **S2** the requester's random numbers
and probe seed are unpredictable and stay secret. **S3** the requester's anchors were computed from the true weights.

- **Claim 1, privacy.** Each provider's rows are jointly uniform over the ring and independent of the input, so one
  provider alone learns nothing about the activations, whatever it computes. This needs no hardness assumption.
  Argument: `x1` is uniform; `x2 = x - x1` is uniform because `x1` is; every other row carries its own fresh uniform
  mask. [test_shares.py](test_shares.py) checks that A's rows are bit-identical for two different inputs under the
  same randomness, and that for B an equally likely choice of randomness gives identical rows for a different input.
- **Claim 2, one dishonest provider is caught.** If one provider changes any value in its real block while the other
  is honest, the attempt is rejected except with probability at most 2^-54 per altered row for the widest small-Qwen
  projection (the run reports the figure as `minimumDetectionLog2`). Argument: to pass, its change to each check block
  must equal `alpha_k` times its change to the real block, and it has never seen `alpha_k`. A change that is a
  multiple of a high power of two is easier to match, but a change that large pushes the result outside the honest
  range, which is checked separately.
- **Claim 3, the same deviation by both providers is caught.** If both providers return zeros, skip rows, or multiply
  by the same other matrix `W + D`, each check is off by `delta_k` times `p D`, which is non-zero unless `D` happens
  to cancel on a probe neither provider has seen. This covers providers that deviate in the same way without
  communicating, for example both skipping the work. Providers that pool their views are outside S1; over repeated
  calls they could also recover the probe.
- **Claim 4, a caught provider learns nothing from being caught.** Rejection is deferred, and whether a check fails
  depends on the providers' own changes and the requester's secrets, not on the input, except through the range
  check with the probability bounded in Claim 2.

## What is not defended

- **Collusion.** If A and B combine their rows they recover every activation, and from those the prompt. Choosing two
  providers under different operators is a placement decision for Reploid and an assumption here, not something the
  protocol can verify.
- **Shape and timing.** Providers see the projection order, the number of rows (the prompt-length bucket), message
  sizes and timing.
- **Model confidentiality.** Weights are public by design.
- **Availability.** Either provider can stop the attempt.
- **Transport and identity.** This prototype talks to providers over process pipes or SSH. Provider identity, grants
  and receipts belong to Reploid and are not built.
- **Flash.** Evaluating only the routed experts would show the providers which experts a token chose. Flash's large
  lookup tables would need a two-server private lookup. Neither is designed.
- **What the requester still stores.** The embedding table (545 MB at 32-bit for small Qwen), normalisation
  parameters, and per projection a few weight-derived vectors (quantization scales, zero points, column sums, bias)
  and the anchor row. These come from the model owner and are trusted like the weights. It stores no weight matrix.
- **GPU execution.** The evaluator is NumPy integer arithmetic on CPU. Doppler's WebGPU kernels are floating point; a
  64-bit ring kernel does not exist yet.

## Measured

Two retained runs of the pinned small-Qwen fixture (four output tokens, eight prefill rows, 169 linear layers per
step), with their [context](evidence/shares/qwen-shares-context.json):

| | [One host](evidence/shares/qwen-shares-local.json) | [Provider A on a second machine](evidence/shares/qwen-shares-two-machine.json) |
| --- | --- | --- |
| Output tokens | 12095, 13, 1084, 374 (" Paris. It is") | the same |
| Largest logit difference from the quantized reference | 0 at every step | 0 at every step |
| Execution | 137.3 s | 167.4 s (not a clean timing, see context) |
| Waiting for providers | 130.9 s | 160.4 s |
| Requester share arithmetic | 0.75 s | 0.71 s |
| Requester non-linear steps and orchestration | 5.6 s | 6.3 s |
| Sent to each provider | 108.6 MB | 108.6 MB |
| Received from each provider | 200.7 MB | 200.7 MB |

The tokens and the plan and model identities are the same as in the retained homomorphic receipts. The
[checked homomorphic run](evidence/integrity/qwen-ntt12-check-local.json) took 89.1 minutes for the same fixture, with
a 268 MB evaluation key; that is a different mechanism with a different guarantee, so the two are shown side by side
and not as a matched benchmark.

The requester stored 544.5 MB of embedding table, 5.6 MB of per-projection quantization vectors and biases, 3.6 MB of
anchors and 0.2 MB of normalisation parameters. Its one-time anchor pass over the public weights took 2.4 s. The
providers held the 169 weight matrices (1.98 GB at 32-bit).

Each decode step took about 13 s. Almost all of it is the providers: five rows of unoptimised 64-bit NumPy arithmetic
per real row, on a two-core laptop processor, with the 151,936-wide vocabulary head the largest single product. Both
runs had one operator and no misbehaving provider, so they show that the route is exact and what it costs, not that
it is private in deployment or that it detects cheating; the tests cover detection.

## Review record

- 2026-11-17, automated tests: 10 pass ([test_shares.py](test_shares.py)).
- 2026-11-17, independent automated review of the first version. It demonstrated that two providers who both
  returned zeros, or both used the same other matrix on every row except a recognisable anchor row, were accepted
  300 times out of 300; that the range check accepted the most negative 64-bit value, letting one provider change a
  zero product about a quarter of the time; that the privacy test never examined B; and that the evaluator would
  open files outside its directory. All are fixed in this version: the probe is now secret and hidden inside every
  check row, the range comparison is strict on both sides, the tests cover each demonstrated strategy, the evaluator
  serves only files inside its directory and hashes the bytes it maps, and a stalled provider is cut off at a
  deadline. The fixed version has not been reviewed again.
- Cryptographer review: none.
