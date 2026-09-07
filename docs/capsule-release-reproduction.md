# Reproduce the installed Capsule release baseline

The release baseline retains an exact runtime archive, signed model metadata,
reference outputs, application assets, and a pinned public-source reconstruction
recipe in `reports/capsule-baseline/20260907-release/reproduction.tar.gz`.
The acceptance receipt in that directory binds the evidence to source commits,
archive hashes, model identities, and commands. It is separate from the earlier
[local physical baseline](capsule-physical-baseline.md) and the npm registry's
different `doppler-gpu@0.6.0` archive.

The archive contains no signing keys or model weight shards. Reconstruction
downloads the exact public Qwen revisions and runs the retained runtime's
converter. Every reconstructed shard must match its signed SHA-256 before it
can enter the retained Capsule. Existing signed manifests, plans, reference
tolerances, approvals, and denials are preserved; reconstruction never signs or
approves a release. A successful restoration is artifact evidence; the following
physical probes establish execution separately.

## Dependencies and isolated checkout

Use Node 22 with its npm and the repository lockfile. The byte-identical local
package build used Node 22.22.1 and npm 9.2.0. Other npm/compression versions can
produce different archive bytes from the same file inventory. Physical evidence
always names the archive actually installed. The replay below installs the
retained archive, not the similarly versioned npm registry package.

For a Linux reproduction, use a writable temporary directory with space for
the downloaded sources, converted shards, installed application, and browser
profile. On the qualification host, `/tmp` exhausted a quota; `/var/tmp` worked.
Keep the source checkout, isolated consumer, and attempt output in separate
directories. Do not run `npm ci` in a checkout while tests use its dependencies.

```sh
git clone https://github.com/clocksmith/doppler.git /var/tmp/doppler-replay-source
cd /var/tmp/doppler-replay-source
TMPDIR=/var/tmp npm ci
npx playwright install chromium
# If Chromium system dependencies are absent on Linux:
npx playwright install-deps chromium

mkdir /var/tmp/doppler-replay-bundle
tar -xzf reports/capsule-baseline/20260907-release/reproduction.tar.gz \
  -C /var/tmp/doppler-replay-bundle
TMPDIR=/var/tmp node tools/restore-capsule-baseline.js \
  /var/tmp/doppler-replay-bundle /var/tmp/doppler-replay
```

Before extraction, compare the archive's SHA-256 with the committed acceptance
receipt. Use the receipt's source revision when reproducing a historical release.
The tool verifies all bundled files, downloads only pinned upstream paths, checks
source byte hashes, converts, and verifies the complete signed artifact closure.
Its new output directory retains the input recipe, commands, logs, downloaded
source identities, and `restoration.json`. Failed attempts remain inspectable;
use another output directory after correcting a failure.

Upstream metadata for both retained Qwen revisions declares Apache-2.0.
The bundle includes that license, upstream model cards, and a modification
notice explaining the derived RDRR artifacts. The Doppler package and reference
application retain their MIT license. Embedded absolute paths in historical
evidence are provenance, not dependencies of this replay.

## Installed Node qualification

The retained Node dependency lockfile pins the public `webgpu@0.4.0` package and
its transitive dependencies by registry integrity. Copy it over the restoration
consumer only after conversion has finished:

```sh
cp reports/capsule-baseline/20260907-release/node-consumer/package*.json \
  /var/tmp/doppler-replay/consumer/
cd /var/tmp/doppler-replay/consumer
TMPDIR=/var/tmp npm ci --ignore-scripts --omit=optional --no-audit --no-fund
cd /var/tmp/doppler-replay-source
```

Write `/var/tmp/doppler-node-qualification.json` with:

```json
{
  "packageBundlePath": "/var/tmp/doppler-replay",
  "capsuleRoot": "/var/tmp/doppler-replay/retained/node-capsule",
  "deniedRoot": "/var/tmp/doppler-replay/retained/denied-capsule",
  "referencePath": "/var/tmp/doppler-replay/retained/references/reranker-reference.json",
  "outputDir": "/var/tmp/doppler-node-qualification",
  "requiredVendor": "amd",
  "providerCreateArgs": ["backend=vulkan", "enable-dawn-features=allow_unsafe_apis"],
  "repeatRuns": 2
}
```

```sh
node tools/qualify-installed-reranker-node.js /var/tmp/doppler-node-qualification.json
```

The probe requires a physical AMD adapter, disables network acquisition, checks
the frozen reranker oracle, destroys and reopens the device, and rejects the
preserved revoked release and rollback. It evaluates release policy using the
current clock and the explicit retained-use record, and copies the checkpoint
ledger into its new attempt directory. It does not mutate the input ledger.

## Installed browser application qualification

Write `/var/tmp/doppler-browser-qualification.json` with:

```json
{
  "applicationDir": "/var/tmp/doppler-replay/retained/application",
  "fixturePath": "/var/tmp/doppler-replay-source/tests/fixtures/document-search-unseen.json",
  "outputDir": "/var/tmp/doppler-browser-qualification",
  "temporaryDirectory": "/var/tmp",
  "launchArgs": [
    "--enable-unsafe-webgpu", "--enable-webgpu-developer-features",
    "--disable-dawn-features=disallow_unsafe_apis", "--ignore-gpu-blocklist",
    "--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface",
    "--enable-precise-memory-info"
  ],
  "requiredVendor": "amd",
  "timeoutMs": 1800000,
  "recovery": true
}
```

```sh
TMPDIR=/var/tmp node tools/qualify-document-search.js /var/tmp/doppler-browser-qualification.json
```

Run Node and browser physical probes sequentially. The browser probe installs
the retained application and actual models, explicitly accepts retained use,
checks the frozen six-query corpus and BM25 incumbent, then closes the browser
and server and repeats search offline. Recovery includes cancellation, quota
exhaustion, corrupted artifact repair, incompatible-index rebuilding, and
physical device loss. These are scoped AMD Node/Chromium results; Bun, generation,
other devices, and external adoption require their own evidence.

Component: doppler.repository-tooling. Intent: preserved.
Acceptance evidence: retained restoration and installed qualification receipts.
Boundary effects: none; application activation and compiler/runtime ownership
remain explicit.

## Rebuild retained application assets without signing custody

Later repository tooling supports `models: null` in
`tools/build-document-search-app.js`, with `previousApplicationDir` pointing to
the reconstructed `retained/application` directory. Keep its `search` and
`storage` settings unchanged, select the installed package bundle, and declare
the desired `loading.maxRetainedArtifactBytes` explicitly. The builder verifies
the signed Capsules and every artifact, preserves their release events and
checkpoints, and rebuilds the application cache manifest. It requires the same
application program. A changed program or model release needs its own reviewed
release workflow; this retained-model mode cannot sign or promote it.

This mode permits an independently reproduced retention experiment using only
the public reconstruction bundle. Unlimited retention remains the default.

## Separately qualified Bun lane

The [Bun acceptance receipt](../reports/inference-coverage/20260907-bun/acceptance.json)
binds signed Qwen reranking, source-reference comparisons, device recovery and
release denials to Bun 1.3.10, the retained runtime archive and physical AMD
Radeon 8060S hardware. It does not qualify generation or another device.

After reconstructing the public baseline and installing its documented consumer,
extract the adjacent Bun `evidence.tar.gz` into a new directory. The archive
includes the exact executed `qualifier-source.js`; its hash matches the receipt.
Rebind `config.json` to the reconstructed paths and a new output directory, then
run `bun qualifier-source.js config.json`. The original configuration and receipt
remain provenance. The newer repository qualifier shares checkpoint and cleanup
helpers, so its source hash differs from this retained historical execution.
