# WGSL Repair V12 Adapter Portability

The three external20 adapters are preserved at revision-pinned Git LFS URLs,
and the reusable PEFT-to-Doppler bridge now reproduces the frozen 628-token
prompt exactly, validates adapter identity, and activates native PEFT LoRA
matrix layouts. Behavioral inference parity is not established.

The frozen V1 gate failed for the base and all three adapters. The base matched
the first token, but its first-token logit cosine was `0.8523495` against the
required `0.995`, with six of ten top tokens overlapping. Every adapter matched
the first token and produced a nonzero effect, but none reproduced the exact
20-token Transformers/PEFT completion:

| Seed | Common prefix | Doppler completion | Logit cosine | Delta cosine |
| --- | ---: | --- | ---: | ---: |
| 11 | 9 tokens | `@group(0) @binding(0` | 0.6646071 | 0.6682266 |
| 29 | 3 tokens | `@group((0) @binding((0) var<uniform` | 0.6769299 | 0.5815704 |
| 47 | 3 tokens | `@group((0) @binding(0) var<uniform> u: Uniforms;;` | 0.6011106 | 0.7239024 |

A float32 Transformers control retained the exact seed-11 adapter completion,
but Doppler logit cosine remained `0.8544648` for the base and `0.6646887` for
the adapter. This rejects BF16-versus-F32 reference precision as the cause of
the failed gate. The remaining boundary is a base-runtime component or layer
divergence on the frozen prompt.

No seed is selected. The V1 thresholds remain frozen, V13 trainer-to-Doppler
parity remains unsatisfied, semantic evaluation remains blocked, and WGSL
Doctor is not authorized. The machine-readable status is
`wgsl-repair-v12-adapter-portability-2026-07-13.json`.

The next diagnostic must capture matched Gamma and Doppler component or layer
boundaries on the frozen prompt and locate the first divergence. A later policy
may be frozen only on unexamined evidence; it cannot erase this failed V1 gate.
