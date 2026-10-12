# Attention residual helper review

This CPU-only review covers the retained layer 15 prefill operands from capture
`4d377a97fd0f1d4aa3f9a4e92900a35fa3b8b33bca0846e72c1e9e85dc731d18` and
the existing helper shader identified in the adjacent JSON. No attention
correction shader existed when this review ran. This is not GPU acceptance or
an independent whole-model reference.

Reproduce on the retained capture:

```sh
node artifacts/source-prefix-correction-20261012/review/attention-residual-domain-review.js reports/local/source-prefix-correction-20261012/rope-frequency-decode7-capture.json
```

The 376,832 Q values are finite normal F32. The 94,208 stored K values are
finite F16, including five F16 subnormals; exact promotion makes them normal
F32. Their minimum binary quanta are respectively 2^-43 and 2^-24. Thus exact
products and nearest-even accumulated residuals are multiples of 2^-67. A
conservative bound on 256 products plus corrections is 124,034.9294. This
excludes F32 underflow and overflow for this captured score-dot domain, assuming
the declared exact F16 promotion. It does not prove a backend's evaluation order.

The existing `addition_residual` is not correct for every finite supplied sum.
For `a=0x00800000`, `b=0x807fffff`, and a sum flushed to `0x00000000`, the
exact residual is `0x00000001`. Its low-word-only reconstruction returns zero.
The sign-reversed case also fails. These are explicitly conditional CPU
counterexamples, not observations that AMD or Metal executed those inputs.
The unflushed nearest-even sum provides a passing control. A general shader
must handle such cases or enforce a bounded domain with preserved behavior
outside it; captured-domain success cannot establish general correctness.

A future attention change must preserve grouped-head mapping (eight Q heads,
two KV heads, dimension 256), causal/window masks and cache offsets, each
declared cache layout, invalid-query barrier participation, score scaling and
softcap, online softmax rescaling, and output normalization. Recovering product
residuals alone does not prove correction accumulation, exponentials, softmax,
or P×V accumulation accurate. These are review requirements, not tests of an
unwritten candidate. The paired downstream intervention and complete frozen
prefix comparison remain separate qualification steps.
