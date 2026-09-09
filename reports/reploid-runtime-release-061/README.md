# Reploid runtime release candidate 0.6.1

This is an unpublished package candidate, based on Doppler
`62475885923e2d376bd7672157dc671bec197451`. The published `0.6.0`
does not export `./generation-contract`, which Reploid requires. A local
archive bearing `0.6.0` is not interchangeable with that registry release.

The candidate increments package and runtime versions together and regenerates
the runtime source closure. Release checks also expose pre-existing shader
digest drift: the generated Glimmer lowering receipt and the two maintained
Qwen reranker recipes now bind the current sampler source. The synthetic bundle
test binds all its temporary kernel references from the canonical registry.
Retained manifests, signed Capsules, physical reports, and support levels are
unchanged. The routing audit retains all existing failures and adds the newly
observed sampler mismatches; checking its freshness does not clear those failures.

Initial full checks failed in semantic lowering, bundle CLI, reranker closure,
and two host-teacher fixtures. The latter two were caused by this isolated
checkout's top-level dependency symlink, which has been replaced with a real
directory. The five focused checks pass; one unavailable historical-evidence
case is explicitly skipped. Initial logs remain separate from repair results:

- `/var/tmp/doppler-061-release-20260908-unit.log`
- `/var/tmp/doppler-061-release-20260908-green-audit-synced.log`
- `/var/tmp/doppler-061-release-20260908-focused-repairs.log`
- `/var/tmp/doppler-061-release-20260908-green-repaired.log`

Publication requires a clean source identity, retained final package, completed
release gates, installed Reploid checks, and authenticated npm publication.
Reploid must then install the exact registry release, regenerate its validation
provenance and browser identities, qualify the installed application, and deploy
matching backend and static bytes. No deployment follows from a package smoke.
Fresh checks on this host returned npm E401; `gcloud` and the usual local Google
Cloud credential files were absent. No credentials are stored in this report.

The separate Gemma grounded-answer development experiment remains on its own
pinned older installed package. Its results cannot certify this candidate.
Likewise, a runtime upgrade does not replace the current public ESM-2 Capsule's
sealed approximate-GELU shader or transfer newer source-equivalence evidence to
that artifact. Numerical and task qualification remain exact-identity gates.

Component: `doppler`, `doppler.runtime-source.config`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: commands and retained logs above; final package and clean
release gate must be recorded before publication.
Boundary effects: package version and future recipe identities; no application
policy, model promotion, signed-artifact rewrite, or production mutation.
