# Reproduce installed generation qualification

This lane compares installed Qwen3-0.6B generation against six frozen PyTorch
source references, including every prompt token and generated token. It also
exercises cancellation, reuse after cancellation, rejection on a destroyed
GPU device, and reopening with a new device. Node, Chromium and Bun run the same
scenario through their installed module URLs and retain separate receipts.
This is raw RDRR model API qualification, not a signed generation Capsule.

Use the repository dependencies and Linux setup described in
[Capsule release reproduction](capsule-release-reproduction.md). The retained
[reconstruction recipe](../reports/inference-coverage/20260907-generation/reproduction.json)
pins the archive, original manifest, upstream revision and license. Its adjacent
`reproduction.tar.gz` contains the runtime archive, manifest, tokenizer, exact
conversion configuration and frozen reference. It contains no weight shards.

Verify the archive SHA-256 against `reproduction.json`, extract it into a new
bundle directory, then run:

```sh
TMPDIR=/var/tmp node tools/restore-capsule-baseline.js \
  /absolute/generation-bundle /absolute/generation-restoration
```

The shared restoration tool downloads the seven pinned public source files,
checks their hashes, invokes the installed converter, and verifies all 25 model
files against the retained recipe. `modelDirectories` explicitly distinguishes
converted shards from retained manifest/tokenizer files. The original manifest
bytes remain fixed; newly generated conversion timestamps and path metadata do
not replace them. The receipt labels these artifacts `raw-rdrr` with
`signedCapsule: false`.

Install the public Node provider into the restoration consumer using the same
committed consumer lockfile as the Capsule baseline:

```sh
cp reports/capsule-baseline/20260907-release/node-consumer/package*.json \
  /absolute/generation-restoration/consumer/
cd /absolute/generation-restoration/consumer
TMPDIR=/var/tmp npm ci --ignore-scripts --omit=optional --no-audit --no-fund
```

Copy the three retained qualification configurations from
`reports/inference-coverage/20260907-generation/inputs/` into your new attempt
area. Change only these path fields:

- `packageBundlePath`: the restoration directory with the installed consumer.
- `modelDir`: its `retained/generation` directory.
- `reference.path`: `inputs/reference.json` in the extracted bundle.
- `outputDir`: a different nonexistent directory for each surface.
- `temporaryDirectory`: a writable temporary directory.

The configuration fixes the source digest, manifest digest, greedy generation,
maximum token count, explicit runtime overlay, repeat count and cancellation
budget. The AMD Vulkan provider configuration identifies this physical lane;
other devices require their own explicit configuration and separate evidence.

From the source checkout, run each surface separately:

```sh
TMPDIR=/var/tmp node tools/qualify-installed-generation.js /absolute/node-qualification.json
TMPDIR=/var/tmp node tools/qualify-installed-generation.js /absolute/browser-qualification.json
TMPDIR=/var/tmp bun tools/qualify-installed-generation.js /absolute/bun-qualification.json
```

Node and Bun use the installed file adapter while rejecting network acquisition.
The browser accepts only the local static server. Receipts retain the exact
runtime archive, source reference, manifest, qualifier/scenario hashes, execution
identity, observed device, raw results, cleanup and memory scope. Successful
reconstruction is separate from successful physical execution. Token equality
establishes equivalence to this frozen source behavior; it is not a general
answer-quality or cross-vendor performance claim.
