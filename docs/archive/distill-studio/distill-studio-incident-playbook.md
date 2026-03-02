# Distill Studio Incident Playbook

## Severity levels

1. `SEV-1`: incorrect claim-driving output, corrupted lineage links, or unverifiable artifacts.
2. `SEV-2`: deterministic replay mismatch or broken compare output.
3. `SEV-3`: degraded diagnostics output, non-blocking docs/tooling drift.

## Immediate actions

1. Capture failing command + report/artifact paths.
2. Run:

```bash
node tools/distill-studio-diagnostics.mjs --report <report.json>
```

3. Run provenance checker directly when needed:

```bash
node tools/verify-training-provenance.mjs --report <report.json>
```

## Containment

1. Freeze external claims tied to affected artifacts.
2. Mark affected report IDs as non-claimable.
3. Gate new outputs until contract/provenance checks return green.

## Recovery checklist

1. Fix root cause.
2. Add regression test.
3. Regenerate artifacts with new report IDs.
4. Re-run release gates.
