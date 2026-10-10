# Frozen reference and first numerical boundary

Component: `doppler.runtime-source.gpu.kernels` and `doppler.tests`.
Intent: preserved. Production runtime, reference inputs, and the 0.001 gate are
unchanged. No package publication or deployment occurred in this investigation.

The retained `0.6.3-dev.split.10` archive reproduces all 55 frozen prefixes exactly
after its first request. That initial request fails separately, including a
maximum logit difference of 67.67570686340332. The complete receipt preserves both
failures and the successful warm repetition. This establishes a retained
reproduction, without claiming the original production record has been recovered.

Native pipeline observation records `main_subgroup` normalization even though
the frozen manifest declares `main`. Changing only that archive's subgroup
registry entry to `main`, with identical shader bytes, makes 16 of 55 warm
prefixes exceed 0.001; maximum difference is 0.0017681121826171875. All 2,793
observed normalization selections then use `main`. This single-variable control
establishes a material reference-contract mismatch; it does not explain every
later arithmetic change or authorize replacing the reference.

The first captured old/current difference is input normalization after identical
embedding operands. Current `0.6.26` selects the declared `main`. On those operands,
an independent Float64 calculation using the exact stored F16 weights and
declared offset improves maximum normalization error from 5.143836663279444e-7
to 4.354780891446808e-7. Public generation observation leaves unmasked logits
unchanged. The first operation's operands are in `normalization-operands.json.gz`.

The existing standard `0.6.26` comparison still fails 56 of 110 frozen checks,
while Mac/Linux unsplit execution and both mixed placements agree exactly on all
55 shared prefixes. Its evidence remains in Reploid's `numerical-026` receipts.

An experimental compensated Q4 projection rounds all 116,736 captured outputs
to the nearest F32 value of the independent Float64 reference on both GPUs.
Its complete native model replay also agrees exactly across the two machines,
but fails 26 of 55 frozen comparisons, maximum 0.002151966094970703. It is retained
as test-only shader evidence and has not replaced the production kernel.

`summary.json` names the exact archives, shaders, hashes, comparisons, interventions,
and local full receipts. Published projections retain per-step hashes, numerical
results, and selected operands; they explicitly identify omitted arrays and
deduplicated sessions. Complete raw receipts remain under
`reports/local/frozen-reference-producer/` and their original physical-run paths.
The earlier trace postprocessing failure and the expected tighter-accuracy failure
of the unchanged Q4 baseline are retained separately from GPU execution outcomes.

Numerical qualification remains failed under the current acceptance contract.
An acceptance change needs explicit authorization: preserve the historical
reference and failure evidence, retain 0.001, and establish a separately named
reference for the corrected declared execution. Do not restore the dispatch bug
or silently regenerate the current fixture.

`proposed-reference.json` describes an inactive candidate prepared from fresh
standard-archive controls using the current manifest on both physical hosts.
All 55 prefixes and the repeated two-prefix request agree bit-for-bit; every
current shader pin matches the archive. The proposed complete reference is
retained locally, with decompressed SHA-256
`a056956131e6944de2ba1d7f13b48327bc9e2c7fc55b55c57f1fc95fcca00046`.
It changes only expected logits. Model binding, prompt tokens, generation,
token labels, stopping labels, precision, and 0.001 remain unchanged. It is
not installed into an executable or acceptance gate, and its agreement does
not establish task quality, lifecycle acceptance, or deployment qualification.
