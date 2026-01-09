# Positive Bias in Hidden States Post-Mortem

**Date:** 2025-12-17
**Status:** DISPROVED
**Model:** Gemma 3 1B Q4_K_M

## Summary

Observed all-positive hidden states at last token position during debugging of garbage output. After extensive investigation, the "positive bias" was revealed to be a **sampling artifact** — debug code only read 5 values, not the full 1152-dim vector. Full vector shows mixed positive/negative signs throughout all 26 layers.

The actual root cause was found in later postmortems: Q4K layout mismatch and missing q_norm/k_norm +1 offset (see GEMMA3-1B-Q4K-GARBAGE-OUTPUT and GEMMA3-QKNORM-OFFSET).

## Initial Observation (Misleading)

```
FINAL_HIDDEN[pos=6]: [183.6, 42.7, 201.1, 63.4, 294.5] - ALL POSITIVE (5 samples)
```

Led to hypothesis that positive bias was accumulating through layers.

## Actual Data (After Fix)

```
LAYER_0_LAST[pos=6]:  min=-10.053, max=9.080   (MIXED)
LAYER_12_LAST[pos=6]: min=-136.646, max=70.206 (MIXED)
LAYER_25_LAST[pos=6]: min=-757.428, max=1421.781 (MIXED)
```

Hidden states have **mixed signs throughout all 26 layers**.

## Bugs Fixed During Investigation

While the hypothesis was wrong, investigation fixed real bugs:

| Bug | Location | Fix |
|-----|----------|-----|
| Attention variant selection | `attention.js:114` | Use `isDecode ? 'decode' : 'prefill'` instead of hardcoded `'prefill'` |
| Workgroup dispatch | `attention.js` | Tier-based dispatch instead of always streaming-style |
| Debug readback timing | `layer.js` | Skip readbacks when using CommandRecorder (batched mode) |

## Key Learnings

1. **Sampling artifacts hide real distribution** — Reading 5 of 1152 values gave completely wrong impression
2. **Position-specific debugging is critical** — Global stats can hide position-specific issues
3. **CommandRecorder timing matters** — Debug readbacks show zeros in batched mode; add `!recorder` checks
4. **Multiple bugs can coexist** — Fixing one bug reveals deeper issues
5. **Wrong hypothesis can still yield useful fixes** — The investigation improved debug infrastructure

## Related Postmortems

- [2025-12-18-gemma3-q4k-garbage-output](2025-12-18-gemma3-q4k-garbage-output.md) — Actual root cause: Q4K layout mismatch
- [2025-12-25-gemma3-qknorm-offset](2025-12-25-gemma3-qknorm-offset.md) — Secondary root cause: missing +1 offset
