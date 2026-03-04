# Training Contract Governance

Operational governance requirements for training/distillation contract claims.

## Required artifacts per release cycle

- Weekly machine-readable contract delta artifact.
- Training contract release-gate pass in CI.
- Distill quality-gate exports (EN and ES) with reproducibility bundle for any
  distill demo claim.

## Commands

- Emit contract delta:
  - `npm run training:contract:delta -- --out ./reports/training/contract-delta/latest.json`
- Run release gates:
  - `npm run ci:training:contract`
- Emit Distill Studio quality gates:
  - `npm run distill:quality-gate -- --report <report.json> --out-dir <dir>`

## Demo claim policy

- Any benchmark or quality claim shown in demo/UI/docs must include a report id.
- Report id must map to a stored report artifact and reproducibility bundle.
- Claims without report ids are out of policy.

## Prelaunch red-team review checklist

- Verify unknown-suite and malformed-training requests fail closed.
- Verify command/suite parity between browser and node surfaces.
- Verify training metrics schema validation rejects incomplete objective payloads.
- Verify forced resume writes explicit `resumeAudits`.
- Verify distill quality-gate EN/ES artifacts are present and reproducible.

## References

- `tools/ci-training-contract-gates.mjs`
- `.github/workflows/training-contract-release-gate.yml`
- `tools/emit-training-contract-delta.mjs`
- `tools/distill-studio-quality-gate.mjs`
