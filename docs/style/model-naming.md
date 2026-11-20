# Model Naming

<!-- style-rules: MODEL-001, MODEL-002 -->

How a model ID is written: the `modelId` in a manifest, `output.modelBaseId` in a
conversion config, and the entries of a catalog. Code that reads a model must not
branch on its name ([General Style Guide](./general-style-guide.md)); the name is
for people and for catalogs, so it has to be regular and it has to be true.

## Characters

Lowercase letters, digits and single hyphens. Nothing else: no underscores, dots,
capitals or spaces, and no hyphen at either end.

```
^[a-z0-9]+(-[a-z0-9]+)*$
```

A dot in a version becomes a hyphen (`3.5` is written `3-5`). A format that GGUF
spells with an underscore has a name token without it; see the table below.

## Shape

```
<family>-<version>[-<release>]-<size>[-<descriptor>...]-<format suffix>
```

| Part | Rule | Examples |
| --- | --- | --- |
| family | the model family, one word | `qwen`, `gemma`, `lfm` |
| version | digits, hyphen for the dot | `3-5`, `3-8`, `2-5` |
| release | the publisher's name for this release, where it has one | `flash-next` |
| size | parameter count; a mixture-of-experts model adds its active count | `2b`, `0-8b`, `270m`, `35b-a3b`, `177b` |
| descriptor | what distinguishes this variant, in the publisher's words | `it`, `instruct`, `text` |
| `source` | the model runs from the original file's tensors, unchanged | `text-source` |
| format suffix | generated from the manifest; see below | `q4k-ehf16-af32` |

The family and the version are separate parts: `qwen-3-8-27b`, not `qwen3-8-27b`.

Every name has a size. Where the publisher's name carries none, it is the parameter
count of the tensors in the manifest, rounded to the nearest billion or million
(`Qwen3.8-Flash-Next` has 176.9 billion, so `qwen-3-8-flash-next-177b`). A codename
is not a name: `flash-peer-effective-q8-v2` says nothing a catalog can sort by.

## Format suffix

The suffix says how the model is stored and computed. It is generated from the
manifest's `quantizationInfo`, not written by hand: `buildVariantTag` in
`src/converter/quantization-info.js` produces the tag the manifest stores, and
`variantNameSuffix` writes that tag in name form. The stored tag is manifest data
and keeps the format's own spelling, so that converting an existing model again
reproduces its identity.

1. The weight format comes first.
2. Then one group for each type that differs from the weights: the letters of the
   parts that use it, then the type. `e` is the embeddings, `h` the output head,
   `a` the activations. Parts stored like the weights are not named.
3. Then, where they differ from the weights: `x` experts, `v` vision, `audio`,
   `tts`, `p` projector, and any per-layer embedding tag.

| Suffix | Reads as |
| --- | --- |
| `q4k-ehf16-af32` | Q4_K_M weights; embeddings and head in F16; F32 activations |
| `q4k-ef16-af32` | Q4_K_M weights and head; embeddings in F16; F32 activations |
| `f16-af32` | everything stored in F16; F32 activations |
| `q40-hq6k-af32` | Q4_0 weights and embeddings; Q6_K head; F32 activations |

### Format tokens

A manifest field keeps the format's own spelling. A name uses the token.

| Format | In a manifest field | In a name |
| --- | --- | --- |
| Q4_K_M | `q4k` | `q4k` |
| Q5_K, Q6_K | `q5k`, `q6k` | `q5k`, `q6k` |
| Q4_0, Q4_1 | `q4_0`, `q4_1` | `q40`, `q41` |
| Q5_0, Q8_0 | `q5_0`, `q8_0` | `q50`, `q80` |
| Float | `f16`, `bf16`, `f32` | the same |

Either spelling is accepted where a quantization is written as input; `q40` means
`q4_0`.

## One name, one identity

A manifest carries its model ID, and a model's identity is the hash of its
manifest. So:

- A name is fixed when the model is converted. Renaming a model is a new
  conversion with a new identity, never an edit to a catalog.
- The suffix must describe the manifest it travels with. A name that says F16
  activations over a manifest that says F32 is a defect, not a nickname.
- A model that should run with different parameters on different GPUs is
  published as one variant per parameter set, each under its own name
  (`...-af16` and `...-af32`). Variants share weights through `weightsRef`; they do
  not share a name.

## Enforcement

- **Characters.** `resolveConvertedModelId` refuses a new model ID that breaks the
  character rule on every path, including a name given explicitly, which is how the
  command-line converter passes `output.modelBaseId` (`assertCanonicalModelId`).
- **Suffix.** A conversion, in Node or in the browser, refuses a name that does not
  end with the suffix its tensors generate, after at least one other part
  (`assertModelIdDescribesFormat`). `tools/refresh-converted-manifest.js`, the tool
  that reissues a manifest, applies both checks to what it writes.
- **Before conversion.** `tests/converter/model-id-naming.test.js` checks every name
  in `models/catalog.json`, and for every conversion config the characters and the
  suffix its own `quantization` block generates. A converter integration test sends
  `new_model` and `a1` through the production path and expects both refused.
- **Not checked by code.** Family, version, release, size and descriptors are read by
  a person. A direct-source runtime label is checked for characters only.

## Identities issued before this guide

These exist and stay valid until their model is reissued. Nothing new may be added
to either list, and the test fails when a listed name is no longer used.

`LEGACY_NONCANONICAL_MODEL_IDS`, names with a character that is not allowed:

| Model ID | What breaks the rule |
| --- | --- |
| `qwen-3-6-35b-a3b-text-source-q4_0-af32` | underscore; the suffix omits its Q6_K head |
| `qwen3-8-27b-text-source-q4_0-af32` | underscore; family and version run together |

`LEGACY_HANDWRITTEN_SUFFIX_MODEL_IDS`, nineteen names whose suffix was written by
hand and is not the one the generator produces:

| Pattern | Names | Generated suffix |
| --- | --- | --- |
| `ehaf16` over a Q4_K_M head and F32 activations | Qwen 3.5 0.8B, 2B and 9B; Qwen 3.6 27B | `q4k-ef16-af32` |
| `ehaf16` over F32 activations, family written `qwen3-8` | Qwen 3.8 27B text | `q4k-ehf16-af32` |
| `ehf16-af16`, the same type written as two groups | Gemma 3 270M, Gemma 4 12B and 31B, DiffusionGemma | `q4k-ehaf16` |
| `f32-af32`, activations named though they match the weights | ESM-2, ESM-C, Nucleotide Transformer | `f32` |
| a descriptor inside the suffix, or parts left out | Gemma 4 E2B, Gemma 4 12B `w4a16-ct` and `hq4k`, the Qwen 3 reranker, GPT-OSS 20B | see the generator |

## Renaming a model

A rename is a reissue: the same weights under a new manifest, with a new identity.

1. **Start from the config the manifest records.** `artifactIdentity.conversionConfigDigest`
   names the version of the conversion config that produced the model; find it in the
   history of the file at `conversionConfigPath`. Copy that version to a new file named
   after the new model ID and change `output.modelBaseId`. Do not start from the config
   as it stands today. Four of the first four models reissued had configs edited since
   their conversion, and one of those edits would have run an F16 head kernel over a
   head stored as Q4_K_M.
2. **Reissue the manifest.** With the weights at hand:
   `node tools/refresh-converted-manifest.js <model directory> --config <new config> --model-id <new name>`.
   Without them, add `--rename-only` and point the tool at a directory that holds only
   `manifest.json`. Rename-only carries over everything the manifest says about how
   the model is stored and run, changes the name, the provenance and a source recorded
   as a local path, and refuses a config that generates a different format or a
   different execution graph from the manifest's own.
3. **Carry the rules.** Rules select behaviour by model ID
   (`src/rules/inference/capability-transforms.rules.json`,
   `src/rules/inference/execution.rules.json`, `src/config/runtime/model-sessions.json`).
   Add the new name wherever the earlier one appears, in the same change, and keep the
   earlier one while its identity is still served.
   `tests/converter/model-id-naming.test.js` compares the transforms the two names
   resolve.
4. **Keep the earlier config and its entry in the lists above** until nothing serves
   the earlier identity.

The weight pack keeps its `weightCapsuleId`, which was formed from the earlier name:
the weights are the same weights.

A direct-source manifest (`metadata.sourceRuntime`) has no conversion config to
refresh from. Its name is the `modelId` field alone, and that is the only field a
rename changes.
