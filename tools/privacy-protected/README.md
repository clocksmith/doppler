# Protected generation research

This JavaScript and WGSL tooling is opt-in. It compiles a bounded CKKS program for
a WebGPU evaluator and exposes a Node key-holder bridge. Reploid owns peer
eligibility and the signed request, transport, and receipt. The evaluator loads
encoded public weights and never loads the secret key. The bridge accepts token
IDs and works without model weights.

## Pinned ticket task

The task emits a support queue and an urgency label for a two-word ticket.
It covers four queues and two priority labels, with both word orders. The 32-word
vocabulary, deterministic train/validation/holdout partition, and answer scorer
live in [triage-task.js](triage-task.js). The holdout contains 64 previously unseen
word pairs in the two orders. The frozen bar is 95% complete answers with at
most a two-point drop from the normal model. It also requires zero out-of-range
approximations and zero non-finite logits on shared-prefix and free-running checks.
This benchmark tests composition of familiar words in a closed vocabulary.

[train-triage.js](train-triage.js) trains the dense causal model and selects the
first passing width from 2, 4, 8 and 16 using validation. The pinned checkpoint
has width 2. The normal circuit uses softmax and GELU. The HE variant uses
a bounded linear attention weight and quadratic GELU. Each component of an
encrypted input is a one-hot slot value. The evaluator folds the public
key–value product into the vocabulary lookup and keeps both encrypted value
moments per position. New decode steps read those ciphertexts at their
original levels. The requester decrypts logits, chooses the next token by
argmax, and encrypts that token for the next step.

```sh
node tools/privacy-protected/pp.js triage-plaintext
node tools/privacy-protected/pp.js gpu-compile --triage --profile research-triage-n14-u32 --positions 2 --decode-through 2 --out /tmp/ckks-triage
```

The Reploid benchmark compares decrypted CKKS logits with the HE-friendly
plaintext circuit on the same prefix. Its tolerance is 0.05 absolute and its
warm target is 5 MB per decode step, including the trap lane and signed frames.

## Parameter screening

The research triage profile uses dimension 16,384, a uniform ternary secret,
and a bounded discrete Gaussian sampler with sigma 3.2 and cutoff 19. The
product of its seven ciphertext and three special primes takes approximately
310 bits. The 2024 guideline table gives 438 bits for dimension 16,384,
uniform ternary secrets, and Gaussian error with sigma 3.19 at a classical
128-bit target. This is a table comparison. Review of the sampler, key switching,
CKKS decryption under adversarial behavior, and the signed protocol remains
open. A known-answer trap measures agreement on the tested inputs.

[Security Guidelines for Implementing Homomorphic Encryption, table 4.2](https://eprint.iacr.org/2024/463)
