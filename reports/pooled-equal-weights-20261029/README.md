# amd-host common-weight preparation and failed numerical gate

Status: **no equal-weight engine win**. These are untimed preparation and
diagnostic receipts, not a passing performance comparison.

The source manifest is pinned to
`b322d7ee4f68387102fde5f7c5049604600101d87c999d9e22ddf19f438af9bc`.
The exporter verifies its shard integrity and reconstructs 321 text tensors into
one canonical F32 parameter set, preserving separate embedding and output-head
values. The converted Doppler artifact matches all 321 tensor hashes exactly.
This is not a comparison of the engines' native Q4_0 and Q4_K_M representations.

Pooled's logical inputs and actual GPU weight buffers, including fused views,
match the declared canonical values or recorded preparation transforms. Its
native CPU embedding table is checked separately. Doppler's GPU materialization
still needs its own complete proof; matching artifact files alone is insufficient.

The first Pooled probe incorrectly used ordinary BPE encoding for special chat
tokens. That failure is retained. The corrected adapter preserves atomic special
IDs and matches Doppler's complete 28-token prompt and first eight generated
tokens. A regression check covers that input boundary.

The frozen full-logit gate fails: among 248,320 first-step logits, maximum absolute
difference is **0.007415771484375**, with 126,151 values beyond **0.001**. Neither
the tolerance nor the token/output requirements were relaxed. Single-token layer
captures show much larger increases at full-attention layers; they do not yet
establish the failing operation. Changing ordinary prefill to Pooled's sequential
path does not remove the cross-engine gap.

The independent rounding experiment uses 190,466 Python `struct` binary16
reference inputs and four range/offset cases for both K and V. Both the native
conversion and explicit nearest-even prototype match exactly. The hypothesis
that the standalone KV conversion caused the mismatch is therefore unsupported.
The prototype remains outside the shipped runtime.

`raw-receipts.tar.gz` retains 66 files: successful and failed probes, complete
logit rows, layer captures, exact source inventories, configurations, source
provenance, conversion output, and independent rounding fixtures. It excludes
model weights, browser profiles, and dependency trees. Archive SHA-256:

`42f1a1250090738927c25a1fcf5b4f28280925021a95988288128d7d58588e0a`

`retained-files.json` records each uncompressed file's length and SHA-256.
The full model remains in amd-host's task-local state directory. Measurements
belong to the frozen sources in the archive, not later checkout revisions.

Next gates: isolate the first attention mismatch, prove Doppler's actual GPU
materialization, validate complete generation and all prescribed logits, and only
then run balanced timing pairs. Transformers.js additionally needs a canonical
ONNX export with exact initializer/state bindings. Metal and D3D12 require
independent measurements; amd-host supplies no claim for either platform.

Component: `doppler.repository-tooling` and `doppler.benchmarks.vendors`.
Intent: preserved. Acceptance evidence: five focused tool tests, conversion
onboarding check, and the scoped physical receipts above. Boundary effects:
new opt-in conversion recipe and unpromoted F32 comparison tools; no runtime
defaults, public model metadata, or performance claims changed.
