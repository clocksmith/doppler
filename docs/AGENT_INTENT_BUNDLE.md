# Agent Intent Bundle Template

Purpose: provide a claims-based change proposal with evidence, so reviewers can validate intent quickly.

See spec: `../../docs/INTENT_BUNDLE.md`

---

## 1) Intent Summary

- Change title:
- Target subsystem:
- Primary goal (latency/throughput/memory/correctness/stability):
- User-facing impact:
- Risk level (low/medium/high):

---

## 2) Contract Changes

- Manifest/config changes:
- Schema updates:
- Rule map changes:
- New/updated tests:
- Invariants preserved:

---

## 3) Change Set

- Files touched (paths only):
- Key diffs (short bullets, no code blocks):
- Deleted/removed behavior:

---

## 4) Evidence Bundle

### Correctness

- Tests run (commands):
- Parity checks (CPU/GPU, dtype parity, kernel parity):
- Numerical tolerances:
- Regression tests added:

### Performance

- Benchmarks run (commands):
- Baseline reference:
- Deltas (ttft, tok/s, VRAM, submit count):
- Profiling evidence (trace ids / perf logs):

### Stability

- Memory pressure behavior:
- Fallback paths verified:
- Error codes exercised:

---

## 5) Constraints and Tradeoffs

- Known limitations:
- Performance tradeoffs:
- Platform impact (Chrome/Edge/Firefox/Safari):
- Feature flags or config gates:

---

## 6) Rollback Plan

- How to revert (paths/flags):
- Safe fallback config:
- Expected side effects of rollback:

---

## 7) Review Checklist

- [ ] Intent matches manifest/config changes
- [ ] Rule maps updated for selection logic
- [ ] Tests cover correctness + parity
- [ ] Benchmarks include baseline and delta
- [ ] No hidden defaults introduced
- [ ] Docs updated (if user-facing)
