# Model Onboarding Playbook

## Goal

Add a new model in a way that is reproducible, debuggable, and honest about what
is actually supported.

This guide is the end-to-end planning document for model onboarding work. Use it
before you touch conversion configs, runtime code, benchmark lanes, or catalog
metadata.

## When To Use This Guide

- You are adding a brand-new model or model family.
- You are extending an existing onboarding from text-only to multimodal.
- You are unsure which settings belong in the conversion config versus runtime
  profiles.
- You need a disciplined plan for research, implementation, debugging,
  verification, and benchmarking.
- You want to avoid repeating the failure modes that showed up in recent Gemma 4,
  Qwen 3.5, and LFM2 work.

## Blast Radius

- Cross-cutting planning guide for conversion, runtime, debug, verify, bench,
  and promotion work

## Why This Exists

Recent history showed the same pattern repeatedly:

- The successful model additions got more explicit over time.
- The broken ones usually failed because a copied assumption landed in the
  conversion config or manifest.
- Performance tuning created noise when correctness had not been locked down yet.

Recent commits that shaped this guide:

- `3f3b15eb`
  Removed execution-v0 and converter auto-generation. Lesson: every new model
  must land on explicit execution-v1 contracts, not hidden runtime synthesis.
- `2894ae60`
  Removed `decodeLoop` from LFM2 and Qwen conversion configs. Lesson: batching
  cadence is a runtime/session concern, not a model contract.
- `5db26e90`
  Fixed Qwen 3.5 config assumptions (`rmsNormWeightOffset`, activation, compute
  precision). Lesson: do not inherit settings from a nearby family just because
  the shapes look familiar.
- `0d37c7d4`
  Fixed Qwen 3.5 Q4K execution graphs and manifests. Lesson: kernel entries must
  match actual weight formats and runtime dtypes, not just pass schema checks.
- `9a478b2c`
  Refined Gemma 4 per-layer inputs and scale handling. Lesson: new architecture
  fields must become manifest/runtime contract fields before performance work.
- `ceff96da`
  Added Gemma 4 multimodal runtime support. Lesson: keep the text decoder green
  first, then add modality-specific execution and tests one slice at a time.

Additional Gemma 3 / Qwen retros that matter:

- `c2d49747`
  Qwen 0.8B needed `chatml`, not the generic Qwen thinking-template flow.
  Lesson: chat-template drift can produce degenerate output even when the model
  loads and the execution graph is fine.
- `88e3f293`
  F16 dequant output precision was compounding across Q4K layers and hurting
  Qwen. Lesson: storage/materialization strategy is part of correctness, not
  just performance, especially for recurrent or linear-attention families.
- `e9f324ce`
  Qwen configs were skipping `postAttentionNorm` despite the weights existing in
  the checkpoint. Lesson: tensor headers beat assumptions copied from another
  family.
- `7ae2dee1`
  Several configs were blocked by null or wrong data fields such as
  `queryPreAttnScalar`, missing `layerTypes`, and mismatched activation dtypes.
  Lesson: manifest refresh failures are often config-data bugs, not runtime
  bugs.
- `04ef183d`
  The manifest patcher originally failed to sync some behavior blocks. Lesson:
  if a refresh tool exists, verify that it updates all behavior-owning fields
  before trusting the refreshed manifest.
- `e087675b`
  Qwen decode-loop tuning came after benchmarking and delivered a clear gain.
  Lesson: decode cadence tuning is a late-stage runtime optimization once the
  model path is already correct.
- `a1c0d7b0`
  Gemma 3 and Qwen paths got more stable once the primary decode/prefill
  kernels lived directly in conversion configs instead of model-ID or
  vendor-specific capability transforms. Lesson: the manifest should describe
  the intended path; capability transforms should only handle genuine hardware
  degradation.
- `30781e5e`
  Inline Q4K capability overrides caused garbled Qwen output and had to be
  replaced with a rule-based mismatch policy. Lesson: do not bypass manifest or
  rule-map ownership with ad hoc JS branches, even when the shortcut seems
  “obvious.”
- `ddae0fdf`
  Gemma compare config and fixture hashes needed refresh after reconversion.
  Lesson: once an artifact changes, compare config, support matrix, and saved
  evidence may all become stale at once.
- `06498818`
  Release policy and local-model fixture checks were tightened after Gemma
  regressions. Lesson: claimable support and release visibility should be gated
  by explicit evidence, not by “it worked once on my machine.”
- `0e028e31`
  TranslateGemma got a focused regression integration test after a subtle Q4K
  regression. Lesson: once a model family burns you, add the smallest durable
  regression test that proves it cannot silently break again.
- `9308b3ed`
  EmbeddingGemma needed LM-head assumptions removed and explicit missing-head
  handling. Lesson: verify the model’s output topology from the checkpoint
  before inheriting causal-LM expectations from the surrounding code.

## Historical Retrospectives

### What Worked

- Verifying behavior directly from source checkpoint headers or framework config
  instead of inheriting assumptions from a nearby model.
- Converting one correct text path first, then layering multimodal support and
  runtime tuning afterward.
- Refreshing manifests from checked-in conversion configs instead of patching
  runtime code around stale artifacts.
- Baking the intended primary execution graph into checked-in conversion config
  rather than depending on model-ID-specific or vendor-specific transform
  branches.
- Adding narrow regression tests tied to the actual failure boundary
  (template, post-attention norm, Q4K path, multimodal token handling).
- Treating benchmark artifacts, compare fixtures, and catalog/support matrices
  as evidence that must be refreshed whenever reconversion changes the artifact.
- Verifying output topology early so embedding-style or modality-specific models
  do not inherit causal-LM assumptions by accident.

### What Did Not Work

- Copying chat-template settings from a nominally similar family without proving
  the target model emits or expects the same reasoning tags.
- Copying normalization, activation, or RoPE settings from Gemma-like models
  into Qwen-like models without checking source headers.
- Treating `decodeLoop` and other runtime cadence knobs as artifact-owned
  settings.
- Letting runtime JS bypass the manifest or rule maps for dtype, kernel, or
  materialization selection because the fast path looked “close enough.”
- Depending on model-ID or vendor transforms to choose the intended primary
  execution graph instead of declaring that graph explicitly in the conversion
  config.
- Assuming a manifest patch or refresh tool updated every behavior block without
  validating the resulting manifest against the config.
- Tuning decode throughput before locking down deterministic text correctness.
- Trusting old compare fixtures or support-matrix rows after a reconversion
  changed manifest behavior or execution graph defaults.
- Assuming every text-family checkpoint exposes a normal LM head and output
  stack.

## Required Touch Points

- Official source material for the model:
  - model card / technical report / framework docs
  - released checkpoint config (`config.json` or equivalent)
  - one trusted reference stack for comparison
- `src/config/conversion/<family>/<model>.json`
- `docs/conversion-runtime-contract.md`
- `src/converter/core.js`
- `src/inference/pipelines/`
- `src/formats/rdrr/classification.js`
- `src/config/runtime/profiles/` when runtime-only tuning is needed
- targeted tests in `tests/config/`, `tests/converter/`, `tests/inference/`,
  `tests/integration/`

Only touch catalog/publication state after correctness is proven.

## Recommended Order

1. Decide the support target before writing code.

   Write down the exact initial claim:

   - existing pipeline or new pipeline
   - text-only or multimodal
   - dense / MoE / linear attention / hybrid attention / recurrent
   - supported quantization formats
   - supported surfaces (`node`, `browser`, or both)
   - correctness target first, performance target second

   If you cannot state the target in one paragraph, the scope is still fuzzy.

2. Build a research packet from source material before editing.

   Collect and save, at minimum:

   - official model card or technical report
   - released checkpoint config
   - exact attention layout
   - RoPE parameters
   - FFN activation and normalization behavior
   - multimodal token IDs and modality limits
   - storage dtypes and quantization constraints
   - one deterministic prompt plus expected behavior from a trusted reference

   Then produce an architecture worksheet with these columns:

   - source field
   - expected value
   - Doppler contract owner
     conversion/manifest, runtime/session, or not yet supported
   - implementation status
   - verification method

   Do not start by copying an adjacent conversion config and editing blind.

3. Classify every setting by ownership before touching the conversion config.

   Treat the settings in three buckets:

   - Manifest-owned / conversion-owned:
     activation, norms, RoPE, layer pattern, token IDs, storage dtypes,
     execution graph, output topology, `textOnly`, modality metadata
   - Runtime-owned / session-owned:
     decode batching cadence, KV layout, KV quantization mode, tracing,
     debug probes, runtime profiles, capability remap policy
   - Evidence-owned:
     benchmark workload, prompt, seed, sampling tuple, warm/cold/load mode,
     compare lane

   Rule: if changing the setting should require re-conversion, it belongs in the
   conversion config or artifact contract. If it should vary per run, it belongs
   in runtime config or benchmark config. If it changes claim semantics, it
   belongs in the benchmark/evidence layer.

4. Choose the smallest correct implementation slice.

   Prefer this rollout order:

   - stage 1: text-only load + manifest integrity
   - stage 2: deterministic text generation
   - stage 3: one modality at a time
   - stage 4: runtime tuning profiles
   - stage 5: claimable benchmark lanes
   - stage 6: promotion/catalog/publication

   For a multimodal model, do not start with image, audio, and video together.
   Keep one green text path first so the debug ladder has a stable baseline.

5. Author the conversion config from the research packet, not from memory.

   Start from the closest existing config, but re-derive every model-owned field:

   - `quantization`
   - `inference.attention`
   - `inference.normalization`
   - `inference.ffn`
   - `inference.rope`
   - `inference.output`
   - `inference.layerPattern`
   - `inference.chatTemplate`
   - `session` baseline only when it is truly artifact-owned
   - explicit `execution` graph

   Rules:

   - Use explicit `null` for disabled nullable fields.
   - Do not leave required fields implicit.
   - Do not put runtime batching policy in the conversion config unless the
     artifact genuinely owns it.
   - Do not hide a model assumption in a runtime profile to avoid adding the
     proper manifest field.

6. Make execution-v1 explicit and boring.

   New model support should begin with the most conservative execution graph that
   you believe is correct.

   Prefer:

   - explicit kernels
   - explicit decode/prefill steps
   - explicit policies
   - explicit session defaults

   Avoid:

   - speculative fused paths as the first implementation
   - capability-specific shortcuts before baseline correctness exists
   - kernel aliases that obscure actual storage dtype or matmul family

   If you are unsure between a fast path and a boring path, ship the boring path
   first and prove the model is correct.

   Pointer from Gemma 3 / Qwen history:

   - the primary path should be declared in the conversion config or rule asset
   - capability transforms should only remove unsupported GPU features or widen
     precision for correctness
   - do not encode model identity or vendor preference in runtime transform code

7. Refresh the manifest and prove artifact integrity before runtime debugging.

   After conversion:

   - inspect `manifest.json`
   - compare manifest fields back to the checked-in conversion config
   - inspect `quantizationInfo`
   - verify tensor role classification
   - verify shard integrity and sampled values when conversion is suspect
   - run `npm run onboarding:check:strict`

   Manifest/config disagreement is not a runtime bug. Re-refresh the artifact.

8. Establish a deterministic text correctness gate.

   Before any tuning work:

   - run `verify` with a deterministic prompt
   - run `debug` when output is wrong, unstable, or suspicious
   - compare one trusted reference output or boundary slice
   - inspect actual emitted text, not just command success

   Treat these as failures:

   - empty or collapsed output
   - repeated token loops
   - obvious template drift
   - NaN-like or wildly unstable logits
   - output that is only “good enough” under one ad hoc runtime profile

   Retro pointer:

   - Qwen and TranslateGemma both needed targeted regression coverage only after
     subtle output mistakes were observed. Add that regression test as soon as
     the failure boundary is known.

9. Use the debug ladder in a fixed order.

   When output is wrong, do not jump straight into kernels.

   Use this order:

   - manifest/config parity
   - tokenization and chat-template parity
   - conversion integrity
   - embedding boundary
   - Q/K/V pre-RoPE
   - Q/K post-RoPE
   - attention output
   - FFN output
   - final logits

   Rules:

   - pick exactly one falsifiable hypothesis at a time
   - run one diagnostic before reading five more files
   - if quantized output is wrong, run an F16/source-precision control before
     changing quantized kernels
   - add permanent probes or trace extensions, not throwaway `console.log`

10. Add multimodal support as separate contracts, not as an afterthought.

   For image/audio/video capable models, answer these separately:

   - are the tensors preserved in the artifact
   - does the manifest expose the modality token IDs and config
   - does the runtime have a real execution path for the modality
   - does the chat formatter produce the correct placeholders and history rules
   - does the modality have its own deterministic verify case

   “Artifact contains tensors” is not the same thing as “runtime supports the
   modality.”

11. Only benchmark after correctness is already clean.

   Bench work starts after verify/debug passes.

   Keep two lanes:

   - parity lane:
     claimable when correctness is clean and fairness inputs are locked
   - throughput lane:
     tuning evidence for engine-favorable decode cadence or phase-specific work

   Required benchmark discipline:

   - keep shared contract and engine overlay separate
   - record prompt, seed, token budgets, cache/load mode, and runtime profile
   - save normalized artifacts
   - treat debug or investigation profiles as non-claimable
   - measure prefill and decode separately

   Do not change kernels to chase a benchmark number if the model has not passed
   deterministic text verification.

   Retro pointers:

   - `e087675b` is the good example: decode-loop tuning happened after the path
     was already benchmarkable.
   - `ddae0fdf` is the warning: reconversion can invalidate compare lanes and
     fixture hashes, so refresh evidence whenever the artifact changes.

12. Promote only after the model is boring.

   A model is ready for promotion only when:

   - conversion is reproducible from a checked-in config
   - manifest matches the config
   - deterministic verify succeeds
   - debug evidence is clean enough that you trust the path
   - Program Bundle export succeeds with a browser/WebGPU reference transcript
   - any benchmark claim points to one saved artifact and one reproducible command
   - a human has reviewed output coherence if the artifact is to be reused or
     published

   Then, and only then:

   - move to external volume / hosted artifact workflow
   - update `models/catalog.json`
   - sync support matrix and related registries
   - run `npm run program-bundle:check` if a checked-in bundle example changes
   - publish hosted subsets

## Verification

Minimum gate for a new model:

- `npm run onboarding:check:strict`
- `npm test -- --help` is not a substitute; run targeted tests that prove the
  new contract
- one real `convert`
- one deterministic `verify`
- one `debug` pass if the model is new enough that you do not yet trust the
  path
- one Program Bundle export when the model is intended for Doe/Cerebras or
  cross-runtime lowering
- browser verification when kernels, attention, cache behavior, or multimodal
  execution changed

Recommended command sequence:

```bash
npm run onboarding:check:strict
```

```bash
node src/cli/doppler-cli.js convert --config '<path|url|json>'
```

```bash
node src/cli/doppler-cli.js verify --config '{
  "request": {
    "workload": "inference",
    "modelId": "<model-id>",
    "modelUrl": "file://<artifact-dir>",
    "runtimeProfile": "profiles/production"
  },
  "run": { "surface": "node" }
}' --json
```

```bash
node src/cli/doppler-cli.js debug --config '{
  "request": {
    "workload": "inference",
    "modelId": "<model-id>",
    "modelUrl": "file://<artifact-dir>",
    "runtimeProfile": "profiles/verbose-trace"
  },
  "run": { "surface": "node" }
}' --json
```

```bash
node src/cli/doppler-cli.js bench --config '{
  "request": {
    "workload": "inference",
    "modelId": "<model-id>",
    "runtimeProfile": "profiles/production"
  },
  "run": {
    "surface": "auto",
    "bench": { "save": true, "saveDir": "benchmarks/vendors/results" }
  }
}' --json
```

Use browser verification as a separate gate when the runtime path materially
depends on browser WebGPU behavior.

## Common Misses

- Copying a nearby model config and assuming norms, activation, or RoPE match.
- Treating decode cadence, KV cache mode, or TurboQuant as model-owned settings.
- Letting model-ID checks, vendor checks, or inline capability probes decide the
  intended primary kernel path.
- Tuning performance before proving deterministic text correctness.
- Calling a multimodal model “supported” because tensors were converted, even
  though only the text path runs.
- Shipping a runtime workaround for a stale manifest instead of refreshing the
  artifact.
- Trusting schema validity more than numeric or output validity.
- Assuming the output head or final projection exists because most neighboring
  text models have one.
- Benchmarking with investigation profiles and then treating the result as a
  release claim.
- Forgetting to refresh compare fixtures, support matrices, or catalog metadata
  after reconversion changes the manifest or execution graph defaults.
- Promoting a model before a human has read the actual output.

## Related Guides

- [composite-model-family.md](composite-model-family.md)
- [composite-pipeline-family.md](composite-pipeline-family.md)
- [03-model-family-config.md](03-model-family-config.md)
- [04-conversion-config.md](04-conversion-config.md)
- [05-promote-model-artifact.md](05-promote-model-artifact.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [08-chat-template-formatter.md](08-chat-template-formatter.md)
- [13-attention-variant.md](13-attention-variant.md)
- [14-quantization-format.md](14-quantization-format.md)
- [15-kvcache-layout.md](15-kvcache-layout.md)
- [../agents/conversion-protocol.md](../agents/conversion-protocol.md)
- [../agents/debug-protocol.md](../agents/debug-protocol.md)
- [../agents/benchmark-protocol.md](../agents/benchmark-protocol.md)

## Canonical References

- [../conversion-runtime-contract.md](../conversion-runtime-contract.md)
- [../getting-started.md](../getting-started.md)
- [../onboarding-tooling.md](../onboarding-tooling.md)
- [../style/general-style-guide.md](../style/general-style-guide.md)
- [../style/javascript-style-guide.md](../style/javascript-style-guide.md)
- [../style/config-style-guide.md](../style/config-style-guide.md)
- [../style/command-interface-design-guide.md](../style/command-interface-design-guide.md)
- [../style/harness-style-guide.md](../style/harness-style-guide.md)
- [../style/benchmark-style-guide.md](../style/benchmark-style-guide.md)
- `src/config/conversion/`
- `src/inference/pipelines/`
- `benchmarks/vendors/README.md`
