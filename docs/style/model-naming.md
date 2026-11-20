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
<family>-<version>-<size>[-<descriptor>...]-<format suffix>
```

| Part | Rule | Examples |
| --- | --- | --- |
| family | the model family, one word | `qwen`, `gemma`, `lfm` |
| version | digits, hyphen for the dot | `3-5`, `3-8`, `2-5` |
| size | parameter count; a mixture-of-experts model adds its active count | `2b`, `0-8b`, `270m`, `35b-a3b` |
| descriptor | what distinguishes this release, in the publisher's words | `it`, `instruct`, `text`, `flash-next` |
| `source` | the model runs from the original file's tensors, unchanged | `text-source` |
| format suffix | generated from the manifest; see below | `q4k-ehf16-af32` |

The family and the version are separate parts: `qwen-3-8-27b`, not `qwen3-8-27b`.

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

- The converter refuses a new model ID that breaks the character rule
  (`assertCanonicalModelId`).
- `tests/converter/model-id-naming.test.js` checks every name in
  `models/catalog.json` and every name a conversion config would emit.

## Identities issued before this guide

These exist and stay valid until their model is reissued. Each is listed in
`LEGACY_NONCANONICAL_MODEL_IDS` and nothing new may be added to it.

| Model ID | What breaks the rule |
| --- | --- |
| `qwen-3-6-35b-a3b-text-source-q4_0-af32` | underscore; the suffix omits its Q6_K head |
| `qwen3-8-27b-text-source-q4_0-af32` | underscore; family and version run together |

Other earlier names use only allowed characters but do not follow the shape or
describe their manifest, for example a family written `qwen3-8`, a suffix written
by hand, or a codename with no family at all. They are corrected the same way: at
the model's next conversion, under a new identity.
