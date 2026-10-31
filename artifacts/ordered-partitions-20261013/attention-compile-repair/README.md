# Paired attention compiler repair

Component: Doppler paired attention. Intent preserved. Boundary effects: no
shape, scratch allocation, dispatch, precision, wrapper or reference change.

The prior three-stage attempt timed out during attention compilation after
first-use normalization/activation/RoPE compilation. intel-host cgroup memory reached
10,950,643,712 bytes; normal cleanup failed. Its receipts remain in Reploid's
`artifacts/recurrent-repair-native-20261013/three-stage/`.

The candidate visits the old maximum and four keys through one paired
exponential callsite, retaining alpha first, key order, zero masking and sum
order. All 32 scalar visibility traces are identical. intel-host's physical native
probe compiles and passes the original three independent attention cases at
unchanged 2e-11 (maximum error 1.083e-15). Queue settlement, device destruction,
host close and exit zero pass. Peak cgroup memory is 2,509,705,216 bytes, with no
limit events. This scoped probe is not a matched compiler performance cohort.

The first diagnostic used WGSL's reserved `active` name; validation rejected
it before dispatch. Its source and failure remain retained. The corrected probe
uses `enabled`. Existing provider caches were retained; candidate shader bytes
are new. Full-model qualification is still open.

All 29 existing Mac browser wrapper cases pass, including immediate/recorded
and borrowed/owned attention outputs, masking and related paired operators.
Kernel registry/generated metadata checks and attention contract controls pass.
The public runtime remains unchanged. The normalization compiler diagnostic is
separate and not part of this scoped source change.
