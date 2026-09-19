# Main integration, 2026-09-19

The user requested direct integration into `main`, without new branches, and
explicitly chose to preserve current main while porting unique old-branch fixes.
[Branch inventory](branch-integration.json) records the exact local and remote
tips checked before publication. All actual remote tips and all local branch tips
are ancestors of the combined main. The stale `origin/dev` tracking ref is not an
existing remote branch. Its history was also reconciled after verifying that all
3,110 tip blobs already occur in main history; no product files changed. Other
worktrees were left untouched.

Historical PR #1 was already closed with its `translate` branch deleted. All
1,450 tip blobs already occur in main history; its commit history was likewise
reconciled without changing product files. GitHub rejected reopening the PR;
no deleted branch was recreated. Its historical closed status is distinct from
the five open PRs integrated through main.

## Reconciliation

- Preserve 0.6.2 Capsule contracts while integrating earlier release history.
- Combine PR #10's shared-device repair with explicit device-scoped caches and
  resource ownership, PR #11's generated interfaces/cache identity, and PR #12's
  declared GPU token selection. Preserve their original physical evidence.
- Port publication-hygiene single-flight device initialization and OPFS queue
  telemetry to the current GPU and storage owners, not the old tooling monolith.
- Integrate local-journey recorder, rotary-frequency and planning repairs, and
  Qwen checkpoint training under the existing experimental support boundary.
- Reconcile patch-equivalent CI/rebase/adapter history without resurrecting
  retired Pack APIs. Every product blob at the rewritten overlay branch tip
  already occurs in main history; only local agent memory is excluded.
- Retain immutable historical model manifests and receipts. Current recipe
  digests and generated registries describe newly built candidates; tests
  reproduce historical receipts with their original declared identities.

Combined validation exposed and repaired missing WGSL language capabilities in
explicit-device pipeline creation. Regression tests prove supported subgroup
compilation survives default-device replacement and missing features reject.
Installed digest checks now hash installed WGSL bytes; a corruption regression
proves the repository digest mirror cannot mask changed package contents.

## Boundaries and remaining recorded debt

One AST dependency model still owns inventories, forwarding edges, dynamic
imports and package resources. Source-description and storage ownership remain
below orchestration. No new inference framework or silent numerical fallback
was introduced. The exact GPU lifecycle soft-line review retains cohesive
cleanup rather than forcing a file split solely to satisfy a line threshold.

The type-debt inventory incorporates 41 pre-existing unchecked implementations
from the merged branches. It does not claim those implementations are strict.
The six strict roots and no-growth gate remain enforced; missing declarations
were supplied, and the RoPE boundary now uses named contracts instead of `any`.
Quarantined Qwen numerical references, adapter artifact codecs and unchanged
snapshot readback are individually inventoried, not allowed as runtime tensor
fallbacks.

## Evidence

[Package identity](package-audit.json) identifies the single archive used by
standalone and Reploid checks. [Acceptance](acceptance.json) records the combined
checks. Historical branch evidence is not relabeled as
acceptance of this new archive. The local retained bundle is
`/var/tmp/doppler-main-integration-20260919`.

Component: `doppler`.
Intent: preserved.
Acceptance evidence: package, branch and acceptance records in this directory.
Boundary effects: GPU/memory ownership, host/inference contracts, preparation
and storage dependency direction, repository checks, and experimental training.
No npm publication, deployment, production signing, model promotion, or Reploid
implementation change is implied by the Git integration.
