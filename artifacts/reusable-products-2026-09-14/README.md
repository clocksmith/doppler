# Reconciled installed consumers: Doppler 0.6.2 candidate

The reconciled Doppler candidate passes standalone installed contracts, the
installed Reploid library provider, remote Doppler CI, and the physical operations
listed below. It remains unpublished. Reploid's complete application suite still
has the two failures recorded below. Neither publication nor deployment occurred.

CI log excerpts remove terminal color escapes and trailing whitespace; original
logs remain attached to the linked workflow runs.

[The machine-readable acceptance record](acceptance.json) binds the runtime,
library, fixture generator, models, environment, and retained CI artifact.

## Exact archives and source

| Item | Identity |
| --- | --- |
| Doppler source | `1af9af2e0c6f378a00d659c444178ac9e6726d51` |
| Reploid library source | `775a7ae0c024a4777349bad83ced6576cc1ab538` |
| CI fixture generation checkout | `7965d41d99df93242acc03361100919724571605` |
| Local Reploid test runner | `a3bf49af2e4831152fc2689bc0f69206344d4de4` |
| Doppler 0.6.2 SHA-256 | `4f9689b173ab80f2d664e7006f99dea98e37eb16e6e9d5b831e2902aee6b91b7` |
| Reploid 0.1.0 SHA-256 | `6f12af6f5d7ea31b23917d35eccc2aceacb29416a07af6441b58048d591f7a46` |

[Download the exact CI artifact](https://github.com/clocksmith/doppler/actions/runs/34897549251/artifacts/10369517086).
Its ZIP SHA-256 is
`dcb47264d0e8bd2a927209492b50d7dcfca7c85a474c32fddc12a5018442cb32`.
Both tarballs also reside in `/home/x/deco/releases/doppler-0.6.2-candidate/`.
The later Reploid runner accepts a supplied archive; it changes no package files.

[Both tarballs reproduce byte for byte](archive-reproduction.json) using Node
22.23.2 and npm 10.9.8. Local Node 22.22.1/npm 9.2.0 produced identical package
file contents in different gzip bytes. The CI tarballs are the accepted candidate.
Published Doppler 0.6.1 is separately identified and tested; it is not this candidate.

## Reusable provider and acceptance

The canonical Reploid provider now consumes public `executeOperation()` and
Doppler's public v2 accumulator. It awaits stable text additions before pulling
another event, verifies completion, and preserves settings, stopping reason,
cancellation, request-bound adapters, and owned-versus-borrowed cleanup. Raw
tokens can precede display text. Published v1 remains explicit compatibility:
the provider emits its verified final text once. Collected results use the same
operation path. Existing `toGenerationRequest` callers retain their old path.
The generated application copy remains byte-identical to the package source.

- [Standalone installed acceptance](ci-standalone.json) and
  [Reploid installed acceptance](ci-reploid.json) pass against the same archive.
  Signed fixtures and injected programs prove contracts, not physical inference.
- [Published 0.6.1/v1 compatibility](ci-published-v1.json) passes and rejects v2
  before execution. [Local exact-archive acceptance](local-exact-archives-reploid.json)
  repeats provider, adapter, cancellation, and reconstruction checks.
- [Actual WebRTC and native IndexedDB replay](ci-webrtc.json) pass with injected
  inference. This is one operator, not independent peers or external adoption.
- [Detached library acceptance](library-package-browser.json) passes 15 checks:
  inert imports, declarations, optional Doppler independence, generated assets,
  non-Doppler browser consumers at root/nested/cross-origin URLs, and Verification
  Worker approval of the provider modules. This report names its npm 9 archive;
  the file-content comparison with the CI library is explicit.
- [Doppler full CI](https://github.com/clocksmith/doppler/actions/runs/34897549306)
  and [local `check:green`](local-doppler-green.log) pass, including 802 test files.

## Physical browser results on the exact CI archives

Chrome 146.0.7680.177 used AMD RDNA 3 with `isFallbackAdapter: false`.
The retained receipts contain complete descriptors, requests, model artifact
hashes, accepted target plans, release policy, and cleanup observations.

| Operation | Standalone | Installed Reploid library |
| --- | --- | --- |
| Qwen3-4B-Instruct-2507 | All 16 frozen reference tokens; `max-tokens` stop | Identical complete output; displayed additions match verified final text |
| Qwen3-Embedding-0.6B | Two finite 1,024-dimensional vectors | Provider exposes generation only |
| Qwen3-Reranker-0.6B Q4K | Two-document operation completed | Provider exposes generation only |

[Standalone receipt](physical-standalone.json) and
[Reploid library receipt](physical-reploid-library.json) bind these results.
Generation text, token IDs, sampling, stopping metadata, and output hash match.
Independent deadlines produce different request and receipt hashes; each consumer
verifies its own execution. Reranking completion adds no new quality oracle.
No new physical shared-device run or GPU performance comparison is claimed.
The earlier shared-device repair and its original physical evidence remain in
the September 12 record under their original archive identity.

## Reproduce and migrate

Check out the recorded source revisions and install their locked dependencies.
Download and verify the CI ZIP and tarball hashes before use. From Doppler:

```sh
node tools/check-packed-package.js --archive /absolute/doppler-gpu-0.6.2.tgz --retain /absolute/new-consumer-bundle
```

From Reploid at the recorded local runner revision, using that new bundle:

```sh
DOPPLER_TEST_REQUIRED=1 DOPPLER_TEST_CONSUMER=/absolute/new-consumer-bundle/consumer DOPPLER_TEST_REPLOID_ARCHIVE=/absolute/reploid-0.1.0.tgz node tests/fixtures/doppler-installed-generation.js
DOPPLER_TEST_CONSUMER=/absolute/new-consumer-bundle/consumer node tests/fixtures/doppler-installed-peer-browser-check.js
```

To repeat physical execution, copy the retained
[standalone config](physical-standalone-config.json) and
[library config](physical-reploid-library-config.json), restore their exact model
artifact closures, and adjust filesystem paths and output directories for the
execution machine. The descriptors retain explicit trust, target-plan and local
release decisions; they do not grant production promotion or current revocation
authority. Run each through `node tools/check-installed-capabilities.js CONFIG`.
The library archive must already be installed in the selected consumer bundle.

Adopt v2 by selecting `doppler.capsule-operation-request/v2` in the provider's
`toOperationRequest`. Append `onUpdate` additions; accept the completed result
only after verification. Do not concatenate cumulative snapshots. Keep existing
v1 requests for published 0.6.1. The package README and Doppler streaming guide
describe cancellation and explicit snapshot compatibility. Transport versioning
does not rewrite signed model identities.

## Preserved failures and remaining release work

[The application comparison](reploid-application-comparison.json) retains current
main, reconciled baseline, and final results. Current main had 59 failed assertions
plus a module-loading suite failure. The reconciled baseline had 24 plus that
suite failure. Behavioral fixture repairs remove obsolete call/comment assumptions
without relaxing production receipt validation or raising the boot budget.
The final local run has 2,537 passing, one failing, and 36 skipped tests. Remote
Reploid CI has 2,536 passing, two failing, and 36 skipped tests:

- Zero's seed contains 133 files against the unchanged 69-file ceiling. Owner:
  Reploid Runtime Configuration, Runtime Host, and Generated Browser Assets.
- The hosted diagnostic fails with `ring reveal phase is not open`. Owner:
  Reploid Poolday provider/coordinator lifecycle. The remote failure remains open;
  a passing isolated rerun is not a repair or established root cause.

Full Reploid application acceptance and deployment remain pending. These failures
do not invalidate the separately passing installed inference provider.

Earlier physical passes use SHA-256 `1214e9ef…`, retained under `superseded/`.
Its longer release notes exceeded the unchanged package budget. The final archive
is 10,906,270 unpacked bytes, below 10,906,744. Historical changelog content was
preserved outside the runtime package. The stale current runtime closure was
refreshed only for the version file; signed model histories were not refreshed.
The npm 9 physical rerun was deliberately interrupted before re-executing the
exact CI bytes. Its partial log remains retained, rather than labeled a model failure.

Publication requires restoring npm authentication (`npm whoami` returned E401).
The candidate, acceptance record, and migration instructions are ready for review.
No release tag, merge, npm publication, deployment, provider superiority, external
adoption, or independent maintainer acceptance is asserted by this record.

Component: doppler; reploid.
Intent: preserved.
Acceptance evidence: exact archives and the linked installed, physical, browser,
reproduction, CI, and failure records.
Boundary effects: the optional Reploid provider consumes Doppler's public operation
contract; the repository harness records the installed library identity.
