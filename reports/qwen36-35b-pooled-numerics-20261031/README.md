# Qwen 3.6 35B-A3B: Doppler vs Pooled numerical differences, explained

Status: **diagnostic**. Untimed numerical probes; no performance claim, no
relaxed tolerance, no runtime or model metadata change.

Lane: Jupiter (AMD Strix Halo, RADV, Chrome 146). Common-weights RDRR
`rdrr-q4_0-pooled-q8-common-v2` (shardSetHash `c6d69d3a…`), built by
`tools/experiments/build-pooled-q8-common-rdrr.js`: the 70 tensors Pooled
requantizes at load hold Pooled's own Q8 values, every other tensor the GGUF's
bytes; both engines' GPU copies of the 70 tensors are verified. Pooled runs with
`attnFlash: false` (F32 KV, as Doppler) unless stated. Doppler source c23a8556,
profile `qwen3-6-35b-a3b-q4_0-pooled-q8-common-diagnostic`.

## Findings

1. **q/k L2-norm form.** Doppler normalizes DeltaNet q/k as
   `x * rsqrt(sum(x^2) + 1e-6)`, the form of Transformers 5.6.2
   `modeling_qwen3_5_moe.l2norm` (FLA). Pooled uses ggml's
   `x / max(sqrt(sum(x^2)), 1e-6)`. They differ where q/k are small, mostly at
   position 0 (one conv tap): Doppler vs Pooled layer outputs differ by
   4e-4 relative L2 from layer 0, router logits by 1.7e-3. A Pooled copy whose
   five L2 lines use the Transformers form agrees with Doppler to 1.9e-7 per
   layer and 2.4e-6 in logits on the same token; short-prompt router logits
   then agree to 2.5e-5. This is a declared semantic difference, not a Doppler
   defect; Doppler keeps the reference form.
2. **Long-prompt drift is router near-tie flips.** On the 2287-token prompt
   every one of 32,917 differing (layer, position) top-8 rows lies in the
   causal cone of three primary flips, each an exact 8th/9th expert swap:
   position 150 layer 26 (margins Doppler 8.5e-5 / Pooled 2.3e-5), position 194
   layer 12 (1.4e-6 / 9.5e-7), position 584 layer 1 (1.3e-5 / 1.3e-5). With the
   L2 form matched, two primary flips remain at margins of at most 7e-6. A
   prefix-length sweep (64..2287) shows discrete jumps at the layer after each
   flip and ~5e-6 agreement elsewhere. Generated tokens stay identical.
3. **Pooled's default path stores K/V in f16.** Pooled with flash vs its own
   F32-KV path differs from layer 3 (the first full-attention layer) and by
   0.24 in short-prompt logits; Doppler is within ~1e-5 of Pooled's F32-KV
   path. Default-path (product) token or logit mismatches are therefore not a
   matched-setting comparison.
4. **Determinism.** Doppler prefill with 256- and 512-token chunks is
   bit-identical; repeated runs are bit-identical. Pooled's `prefillUbatch`
   had no effect in this harness (256 vs 512 bit-identical, flash on or off),
   so no within-Pooled summation-order control was obtained.

## Gate under the owner-forwarded route-flip revision

End-to-end part (exact greedy tokens; 0.001 logit tolerance only at positions
outside every primary flip's cone; flips counted with margins), on the c3
logit capture (128 tokens, 26 logit positions per prompt): short prompt passes
(tokens identical, worst 1.12e-4, no flips); long prompt passes on tokens, with
all 26 logit positions excused (0.13 to 0.84) by the three flips above.

Not covered: the CPU-reference half (no independent reference decodes these
Q4_0/Q4_1/Q8_0 tensors; the existing one reads Q4_K_M), the per-layer router
kernel gate, and decode-time routes of generated positions.

## Evidence

External, under `deco-evidence/doppler-35b-numerics/`: `c3` logit capture,
`c4`/`c5` single-token layer probes (flash off/on), `c7` length sweep, `c8`/`c9`
chunking controls, `c10` route probe, `c11` two-token prefill probe, `c12a`/`c12b`
Pooled flash controls, `c13` Pooled with the Transformers L2 form
(`doppler-work/pooled-l2hf`, five lines changed, marked `DIAGNOSTIC-L2HF.txt`).
`retained-files.json` lists SHA-256 of the key receipts. The probes
(`LAYER_PROBE`, `PREFIX_PROBE`, `ROUTE_PROBE`) and analysis scripts live in the
out-of-repo harness `doppler-work/perf-x`.

Open lead: the Qwen 3.5 0.8B equal-weight gate failure
(`reports/pooled-equal-weights-20261029`) runs the same DeltaNet layers; the
L2-form difference is a testable contributor there.

Component: `doppler.benchmarks.vendors` (report only). Intent: preserved.
Acceptance evidence: the receipts above. Boundary effects: none.
