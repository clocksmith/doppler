# Training Contract Governance

Governance policy references the canonical runbook in [training-handbook.md](training-handbook.md).

## Required Artifacts Per Release Cycle

- contract-gate pass
- workload-registry verification
- report-id publication artifact
- compare and quality-gate artifacts for claimable lora or distill outputs

## Governance Rules

- workload packs are the source of truth for behavior-changing operator policy
- run-root artifacts must preserve workload, dataset, and surface traceability
- browser surfaces must fail closed for unsupported training operator commands
- claim publication requires deterministic traceability fields and reproducible artifacts

## Commands

Use the canonical command list in [training-handbook.md](training-handbook.md#primary-commands).

## References

- [training-handbook.md](training-handbook.md)
- [training-claim-traceability.md](training-claim-traceability.md)
- [training-artifact-policy.md](training-artifact-policy.md)
