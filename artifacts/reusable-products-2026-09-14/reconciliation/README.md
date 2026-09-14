# Current-main reconciliation follow-up

Reploid main advanced during acceptance. Merge `7b0258a3` incorporates main
`e4c12cee`, including its bounded Zero boot and hosted library module-loading
repairs. The merge preserves the stronger connection cancellation and cleanup
already present on the integration branch. Generated manifests were rebuilt from
canonical source, and module metadata retains forwarded dependency declarations.

The public Reploid package and generated library copy are **unchanged** from the
previously accepted source `775a7ae0`. The updated required workflow pins the
reconciled consumer and reproduces both exact archives:

- Doppler 0.6.2 SHA-256: `4f9689b173ab80f2d664e7006f99dea98e37eb16e6e9d5b831e2902aee6b91b7`.
- Reploid 0.1.0 SHA-256: `6f12af6f5d7ea31b23917d35eccc2aceacb29416a07af6441b58048d591f7a46`.

[Required installed CI passes](https://github.com/clocksmith/doppler/actions/runs/34900241483).
[Its retained artifact](https://github.com/clocksmith/doppler/actions/runs/34900241483/artifacts/10370796246)
has ZIP SHA-256 `bb04d187410b6a715ec2e23124fe422519f4f13d9ec07fc57ee87b7c8b38de7b`.
Fixture generation used `f6714c9032c899f4ac8505a9c695b6b46e815ce0`.
The original physical receipts still identify the exact runtime and library
bytes used here; no additional physical inference run was necessary.

[Local application tests](local-unit.log) pass: 2,546 passed, zero failed,
36 skipped. [Browser boot and peer-room tests](browser-boot-peer.log) pass 55
cases with five explicit scenario skips. Four of those skipped cases were then
[run in server-relay mode](browser-relay-recovery.log), all passing: interrupted
publication, polling failure, rate limiting, and offline/online recovery. The
remaining skip is the optional twelve-provider scenario. These browser tests
use SwiftShader and synthetic inference receipts; they prove application behavior.

The first remote run passed all 2,546 tests, then its module-system verifier
found a missing generated VFS entry for `doppler-stream.js`. Local continuation
also exposed the Change Passport SDK bundle still containing pre-extraction
owners. Commit `0c1faa03` regenerates both artifacts and the browser manifest.
The [SDK regression tests](sdk-tests.log) and remaining module, registry,
runtime-config, browser-bundle, SDK, type, and Pool configuration checks pass.
Pilot structural validation still reports its unfrozen external fields.

[The final Reploid workflow](https://github.com/clocksmith/reploid/actions/runs/34900703396)
records acceptance of these generated repairs. The prior reveal-phase failure
remains preserved: passing local and remote runs do not establish its root cause
or eliminate its observed intermittency. Zero's earlier seed-size failure is
resolved by the merged upstream repair without increasing its limit.

[Doppler full CI also passes](https://github.com/clocksmith/doppler/actions/runs/34900241482),
including repository, WebGPU kernel, demo controls and offline checks.

[Machine-readable follow-up](acceptance.json) and the parent record bind exact
source revisions, package identities, results, and historical failures.
Publication still requires npm authentication. Neither deployment nor independent
adoption is claimed.

Component: Reploid Runtime Configuration, Runtime Host, Generated Browser Assets,
repository tooling, and Doppler repository tooling.
Intent: preserved.
Acceptance evidence: linked installed CI, browser, unit, SDK, and contract checks.
Boundary effects: current main application repairs reconciled; public library and
Doppler runtime bytes unchanged.
