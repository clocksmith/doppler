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
manifest's `quantizationInfo`. The converter uses `buildModelNameSuffix` for
the name and `buildVariantTag` for the stored tag. Existing manifests retain
their original tag spelling, including grouped roles and GGUF underscores.

1. Write the weight format and any layout first.
2. Group embeddings (`e`) and the output head (`h`) by storage type when it
   differs from the weights.
3. Write the declared activation precision in its own part, such as `af16` or
   `af32`, even when it matches the weights or embeddings. If the source
   declares no activation precision, the converter leaves that part absent.
4. Append experts (`x`), vision (`v`), audio, TTS, projector (`p`) and the
   per-layer embedding tag where the format record calls for them.

| Suffix | Reads as |
| --- | --- |
| `q4k-ehf16-af32` | Q4_K_M weights; embeddings and head in F16; F32 activations |
| `q4k-ef16-af32` | Q4_K_M weights and head; embeddings in F16; F32 activations |
| `f16-af32` | everything stored in F16; F32 activations |
| `f32-af32` | F32 weights and F32 activations |
| `q4k-ehf16-af16` | Q4_K_M weights, F16 embeddings and head, F16 activations |
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

`LEGACY_HANDWRITTEN_SUFFIX_MODEL_IDS` identifies exact older IDs that callers may
still request explicitly. This includes Janus 1B, the Qwen 0.8B canonical F32
config and the Qwen 3.6 27B F16 variant, which were issued under the earlier
grouping rule. An explicit request for an existing ID retains its identity.

### Catalog audit (2026-11-20)

Across 31 Doppler catalog entries, 23 already end in the suffix their format
records generate. This includes the F16 Gemma variants with matching F16 heads
and the three F32 protein/nucleotide models. Eight have a suffix or descriptor
placement that would change under a reissue.

| Issued models | What a reissue must express |
| --- | --- |
| Qwen 3.5 0.8B and 2B | Reploid already carries `q4k-ef16-af32-vf16` and `q4k-ef16-af32` respectively. The Doppler entries resolve older manifests. |
| Both Qwen 3.6 27B variants | Their heads use Q4K and their embeddings use F16. The two suffixes are `q4k-ef16-af32` and `q4k-ef16-af16`. |
| Three Gemma 4 E2B variants | After the text suffix, include `vf16-audiof16-pf16`. Preserve `int4ple` for the two variants that declare per-row embeddings. |
| Gemma 4 12B compressed-tensors W4A16 | Place the `ct` descriptor before the generated `w4a16-ehf16-af16` suffix. |

Doppler’s existing 0.8B and 2B IDs and short aliases keep their original
Hugging Face manifests. The reissued IDs have separate catalog records. Their
`hostedSource` pins the exact manifest URL and SHA-256 and gives the Hugging
Face revision and directory for the payload files. The quickstart loader reads
the manifest from the first source and the tokenizer and shards from the
second. The legacy data directory also contains a different manifest.

Quickstart promotion requires an inference receipt for the reissued identity.
For a local verification run, give the harness a `file://` RDRR directory
containing the reissued manifest and matching payload files. A manifest hash
mismatch fails before model loading.

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
