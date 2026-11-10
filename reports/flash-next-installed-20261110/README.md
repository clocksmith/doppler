# Flash-Next installed candidate: configuration matters

The private package built from `6da05a25` reproduces the earlier bounded
decode result when given the **same explicit prepared plan**. All six retained
F32 score vectors are byte-identical to the frozen runtime: 1,489,920 values,
zero violations under the unchanged `quantized-v1` policy. This is Node on the
physical AMD RDNA-3 device, with F32 KV storage; it does not qualify the model
generally.

| Installed configuration | Archive revision | Short prompt: prefill + 2 decode steps | Long prompt: prefill + 2 decode steps |
| --- | --- | --- | --- |
| Named `flash-next-routed-f32kv-qualification` profile (before `e1a02e41`) | `6da05a25` | Pass | Fail: 69,941 / 128,403 / 102,431 violations |
| Explicit preceding prepared plan | `6da05a25` | Pass | Pass: zero violations in all three vectors |
| Named `flash-next-routed-f32kv-qualification` profile (repaired) | `e1a02e41` | Pass: zero violations (byte-identical) | Pass: zero violations in all three vectors (byte-identical) |

Both configurations produce all six expected greedy tokens. Their complete
score vectors establish different outcomes. The prepared long decode step 2
has maximum absolute error 0.0623848438 and passes the original mixed absolute
and relative tolerance; this is not a claim that every error is microscopic.

The named configuration follows the artifact's compiled plan; the preceding
prepared plan overlays prefill control projections and routing with subgroup
matmul bindings. The overlay repeats seven roles in the base and thirteen
layer-specific blocks. Its 98 serialized binding changes are an explicit plan
difference, not 98 demonstrated defects. This comparison does not isolate
which bindings cause the numerical difference.

[receipt.json](receipt.json) binds the `6da05a25` private archive, source,
installed-file inventory, consumer, inputs, reference, and both observations.
All 1,932 installed files were verified against the tarball inventory before
loading. [prepared-vector-identity.json](prepared-vector-identity.json) records
the six payload hashes checked against the original observation.
[named-profile-comparison.json](named-profile-comparison.json) retains the
`6da05a25` failure; [prepared-plan-comparison.json](prepared-plan-comparison.json)
retains the `6da05a25` bounded pass.

After `e1a02e41` added the seven prefill subgroup-projection bindings to
`profiles/flash-next-routed-f32kv-qualification`, a fresh private archive built
from `e1a02e41` (`47c1d7a1fba60922b7a851e4ed16d30b9d6132975f46cd0c6f6fd840d28f7b7e`,
1,932 installed files verified before loading) was evaluated in an isolated
consumer directory using **only** `runtimeProfile: "profiles/flash-next-routed-f32kv-qualification"`.
[repaired-named-profile-receipt.json](repaired-named-profile-receipt.json),
[repaired-named-profile-comparison.json](repaired-named-profile-comparison.json),
and [repaired-named-profile-vector-identity.json](repaired-named-profile-vector-identity.json)
record zero violations across all six score vectors (`1,489,920` logits) under
unchanged `quantized-v1` and byte-for-byte SHA-256 identity with all six
retained prepared-plan vectors. Original private observations and historical
`6da05a25` receipts remain unchanged.

The public compatibility consumer uses a separate package directory. Its
native Node WebGPU dependency was already provisioned; this is **not** a clean
native-provider installation test. The archive is a private candidate with
source version 0.6.36. No npm version was overwritten, package published,
catalog entry promoted, or service deployed.

The [new inventory prompt failure](../flash-next-new-prompts-20261110/README.md)
remains unresolved even on the preceding prepared plan. Longer generation,
browser and other device coverage, F16 cache, and a reproducible intended
release configuration remain separate work. A passing package smoke does not
close those numerical requirements.

Component: `doppler.tests`, `doppler.repository-tooling`. Intent: preserved.
Boundary effects: installed-package evidence only; no runtime or acceptance change.
