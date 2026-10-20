# Per-tensor representation qualification — 2026-10-17

The loader implementation is on main through `d7abab12`. The frozen candidate
passes the available physical matrix: **12 model journeys and two GPU fixtures**.
**These model profiles show no GPU-memory reduction and no established speedup.**
M3 27B is unexecuted; the full requested matrix remains incomplete. The hosted
GPU-training bootstrap failure and its skipped checks remain open. No release
promotion, quality acceptance, competitor win or external adoption is claimed.

## Implementation and identity

Requirements are compiled before weight reads from each bound tensor, exact
layer and phase, composite consumers, physical aliases, tied endpoints and
reachable execution-plan fallbacks. Q4 consumers retain packed, F16 and F32
representations independently. A stored-F16 tensor under an adaptive Q4 path
retains its F16 primary and any required F32 companion. Unknown consumers remain
conservative. Legacy callers without resolved paths retain their existing policy.
Completed requirements survive legacy downcasting; failed preparation releases
allocated copies. Alias analysis does not add allocation deduplication. This is
not universal optimization of every quantization format.

Implementation commits: `620d1fbb`, `fdfc31ce`, `d7abab12`.

- Unchanged baseline source: `bcb6c045`; [archive](../qwen-sampling-correction-20261017/doppler-gpu-0.6.32.tgz).
  SHA-256 `75f3a81f032e3a0f89102fb9a50e0b202657f69eab71569585d7ffe4ce232bb2`.
- Final candidate: `candidate-v3/doppler-gpu-0.6.32.tgz` inside the raw bundle.
  SHA-256 `f3e9bdef19cf4986c2a0fb4ee8242bdf6f71185acc44d3259d59ae23e1922901`.
- Candidate package: **1,961 files / 3,487,727 packed bytes / 14,154,373 unpacked
  bytes**. Original limits remain 2,004 / 3,497,999 / 14,172,897. The generated
  rule bundle uses compact JSON; readable rule assets remain canonical.
- The exact archive passes installed-consumer smoke, including 32 public export
  types. Focused loader/planner/cleanup/alias/fallback tests, strict types,
  architecture, style, rule sync, public boundaries and dependency checks pass.
  Historical failed iterations remain in `checks/`; the final three-file loader
  regression is `checks/doppler-plan-adaptive.log`.

No model weights, production settings, kernel arithmetic or public services were
changed. Component: loader representation planning/materialization and its
qualification tooling. Component intent is preserved; ownership boundaries are
unchanged.

## Physical comparison

Exact models: `qwen-3-5-2b-q4k-ehaf16` and
`qwen3-8-27b-text-q4k-ehaf16`. Q4_K weights, F32 compute, F16 KV, 2,048-token
context. The 27B uses mandatory 256-token prefill chunks for this campaign.
Apple M3 (24 GiB, Metal, Chrome 155.0.8059.80) runs 2B; Radeon 8060S
(RADV STRIX HALO, Vulkan, Chrome 146.0.7680.177) runs both models. M3 27B was
not provisioned, and 12 GiB free disk was below the approximately 16 GiB artifact.

Each host/model uses baseline → candidate → candidate → baseline. Each invocation
opens a fresh browser and model, then executes one first request, three warm
requests, full public prefill logits, an 833-token long prefill, cancellation,
after-cancel generation, destruction immediately after a GPU submission, device
recreation, regeneration and unload. Short requests use the raw five-token prompt
`The capital of France is`, greedy selection and up to 32 output tokens.
All served JS/JSON/WGSL files must match the immutable archive inventory.

The following values are medians of two invocations; warm values first take the
median of three requests per invocation. Both orders and every sample remain in
the [per-invocation table](comparison-table.md) and [machine-readable summary](summary.json).
No outliers were removed.

| Hardware / model | Engine | Opening ms | Warm TTFT ms | Decode tok/s | Peak summed RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| M3 / 2B | baseline | 4487.6 | 120.5 | 43.07 | 3024.9 |
| M3 / 2B | candidate | 4012.4 | 112.4 | 43.44 | 3267.8 |
| 8060S / 2B | baseline | 5878.4 | 67.6 | 81.16 | 2024.9 |
| 8060S / 2B | candidate | 5931.6 | 71.0 | 80.34 | 2053.2 |
| 8060S / 27B | baseline | 52839.6 | 284.6 | 8.84 | 1610.9 |
| 8060S / 27B | candidate | 53587.7 | 273.2 | 8.82 | 1652.2 |

Both engines retain **1,982.2 MiB** of model buffers for 2B and **16,214.1 MiB**
for 27B. Peak pool allocation is likewise identical: **2,612.0 MiB** and
**17,539.2 MiB** respectively. The candidate plans 151 packed-only plus 170
dense-only tensors for 2B, and 401 packed-only plus 450 dense-only tensors for
27B. The baseline already keeps these Q4 profiles packed. No model shadow
removal is demonstrated here. AMD candidate opening is slightly slower in this
sample; the M3 opening difference depends strongly on order. Do not infer a
performance improvement from two invocations or from small decode differences.
The location/planning phase rises from 0.7 to 3.25 ms on M3 2B, 1.2 to 5.75 ms
on AMD 2B, and 2.2 to 12.4 ms on AMD 27B; it includes location construction.

The separate two-layer GPU fixture proves the conditional saving on both devices:
legacy role-wide loading allocates **1,536 bytes**, while per-tensor loading
allocates **1,024 bytes**, removing exactly one unnecessary 512-byte F16 shadow.
An independent all-one numerical oracle checks packed bytes and F16/F32 values.
Fallback and embedding/head-alias cases retain all three representations; a
stored-F16 adaptive matmul returns exactly 256 while preserving its F32 companion.
This fixture is mechanism evidence, not a full-model percentage saving.

Every compared short/long token sequence and full prefill-logit byte hash matches
within its host/model. Cancellation and recreated-device generation reproduce
prior output. After unload, all runs report zero retained-model, active and
deferred-cleanup bytes; small reusable pools remain. These are workload-scoped
checks, not general runtime correctness or acceptance of model answer quality.

Opening excludes browser launch and downloads and does not purge OS/driver
caches. Recreated-device opening, first request, warm TTFT, long prefill, complete
latency and decode are separate fields in the summary. RSS samples sum this
browser's processes every 500 ms; shared mappings may be double-counted and short
peaks missed. GPU numbers are Doppler pool counters, not vendor-wide VRAM.
Raw `ownedWeightBytes` sums a tracking set that can include retired buffers and is
not used as resident memory. Loader source-read counters are not downloaded bytes.
Hosts were not exclusive. There is no separate CPU-time measurement or arbitrary
spontaneous-device-loss qualification.

## Integrity, retained failures and open gates

Every shard and tokenizer file passes the pre-existing, commit-pinned Reploid
SHA-256 piece indexes: 33 files for 2B on each host and 255 files for 27B on AMD.
The 2B whole-file hashes also agree across hosts. The bundle includes both indexes,
[their provenance](model-piece-provenance.json) and verification receipts.
An independent standard-BLAKE3 check of the legacy 2B artifact disagreed with its
manifest; Doppler's existing hash implementation reproduced that legacy field.
That separate compatibility discrepancy is retained, not repaired or silently
accepted as standard BLAKE3. The independent SHA-256 indexes establish the bytes
used here. No artifact was rewritten.

Earlier failures remain: missing snapshot provenance, an undersized 193-token
harness context, querying the pool after device destruction, and attempts to use
an obsolete Qwen3.6 27B manifest. Its attempted metadata refresh occurred only in
report copies and failed; final 27B cases use the current provisioned Qwen3.8
artifact. Earlier candidate packages/receipts remain separate from final-v3.

[CI run 39081637450](https://github.com/clocksmith/doppler/actions/runs/39081637450)
failed one of 643 test files at `tests/training/native-lora-round-gpu.test.js`:
`native_training_webgpu_bootstrap_failed` (no usable authorized adapter; Vulkan
initialization failure). Node WebGPU/browser units, packed-package and demo/PWA
steps were skipped. Local focused checks do not close those hosted gates.
The full M3 27B lane also remains open. Resolve provisioning/disk capacity before
running it. Repair the hosted bootstrap in its own scope, then actually execute
its skipped checks. Do not divert this checkpoint into LoRA, WAN or prefix work.

## Evidence and reproduction

[Raw evidence](raw-evidence.tar.gz) retains packages, inventories, all physical
receipts and logs, old failures, integrity indexes and check/CI logs.
[SHA256SUMS](SHA256SUMS) pins the bundle and readable control files.
[The frozen contract](campaign-contract-v3.json) and [driver identities](driver-identities.json)
pin the final comparison. Extract the bundle into this report directory before
reducing or inspecting bundled paths:

```bash
tar -xzf reports/weight-representation-20261017/raw-evidence.tar.gz -C reports/weight-representation-20261017
python3 reports/weight-representation-20261017/summarize.py
```

For a new run, use already-provisioned exact model bytes, verify them with
`verify-model.py MODEL_DIR PIECE_INDEX OUTPUT_JSON`, and extract each immutable
package with `extract-runtime.py ARCHIVE NEW_RUNTIME_DIRECTORY`. Copy the campaign
contract, set that host's model/runtime paths and a fresh output directory, remove
the historical `waitForReceipt` field, then execute
`run-campaign.py metal|vulkan COPIED_CONTRACT`. The runner refuses to overwrite an
existing case. Preserve the new contract and receipts separately; adapt the
reducer's contract path for that run. Never benchmark mutable working-tree files.
