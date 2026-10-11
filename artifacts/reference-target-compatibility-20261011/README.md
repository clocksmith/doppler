# Numerical target compatibility

The unchanged historical capture and the independently evolved F64-equation /
F16-cache target cannot both be satisfied at absolute tolerance 0.001 for all
55 prefixes. A fresh offline comparison of the hash-verified full vectors finds
174 incompatible logits across three prefixes. No GPU, kernel, weight, precision
policy, tolerance, or reference authority changed.

At request 2, step 28, vocabulary index 106534:

- Historical target: **-6.299257278442383**.
- Independent target: **-6.302488761834255**.
- Separation: **0.0032314833918718477**.
- Best possible maximum error to both: **0.0016157416959359239**.

By the triangle inequality, a value within 0.001 of each target would require
the targets to be at most 0.002 apart. Their admissible intervals are disjoint.
This is a compatibility result, not evidence that either target is automatically
authoritative, that current runtime accuracy is acceptable, or that the user
previously required passing both: F64 remains a diagnostic proposal. It rules
out treating both as simultaneous 0.001 release requirements.

The historical producer used an undeclared normalization selection. Restoring
that selection solely to imitate its output remains inappropriate. Independently
specified mathematical accuracy and conformance to a declared F32 evaluation
also remain different contracts. The reference decision is pending explicit
approval under canonical TODO ALL-4. Neither approving a reference nor this
incompatibility result turns the existing failed candidates into passes.

[check.py](check.py) verifies all retained vector shards, evaluator/contract
identities, coverage, finite values and the original historical archive before
comparing all coordinates. [result.json](result.json) retains the witness and
each prefix. Reproduce from the Doppler root on the host retaining the indexed
vectors:

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=1 \
  python3 artifacts/reference-target-compatibility-20261011/check.py
```

Component: independent numerical evidence. Intent: preserved. Boundary effects:
none. No acceptance reference is activated and no package is promoted.
