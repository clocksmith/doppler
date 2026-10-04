# Local model metadata and tokenizer custody

Tracked manifests and origin records retain their existing model identities.
Byte-identical tokenizers share an original through relative symlinks within
this tree. Tokenizer filenames, resolved bytes, and declared hashes are unchanged.
Unique tokenizers remain regular files.

## Shared originals

| Tokenizer group | Original relative to this directory |
| --- | --- |
| Gemma 4 and converted DiffusionGemma JSON | `gemma-4-e2b-it-q4k-ehf16-af32/tokenizer.json` |
| Gemma 3 270m and TranslateGemma JSON | `gemma-3-270m-it-q4k-ehf16-af32/tokenizer.json` |
| Shared Gemma SentencePiece model | `gemma-3-270m-it-q4k-ehf16-af32/tokenizer.model` |
| Qwen 3.5 and 3.6 JSON | `qwen-3-5-0-8b-q4k-ehaf16/tokenizer.json` |
| Gemma 3 1b JSON | `gemma-3-1b-it-q4k-ehf16-af32/tokenizer.json` |
| Glimmer text JSON | `muse-glimmer-30b-text-f16-af32/tokenizer.json` |
| Qwen reranker JSON | `qwen-3-reranker-0-6b-f16-af32/tokenizer.json` |
| GLM OCR JSON | `glm-ocr-f16-af32/tokenizer.json` |

Grouping reflects exact file bytes, not equivalent tokenization or model support.
Preserve these originals when retaining the linked model directories. Check out
Git symlinks as links to use this developer-local mirror.

## Independent copies

Dereference links when exporting a model directory outside this tree. For example,
from the repository root:

```bash
cp -RL models/local/qwen-3-reranker-0-6b-q4k-ehf16-af32 /destination/model
```

This copies the available files, including the actual tokenizer bytes. Model
weights still need to be acquired if they are absent from the local directory.
Rig materialization uses `copyFile` to produce independent Capsule artifacts.
Prepare an independent copy before revising a tokenizer for one model variant.

This retention change reduces checkout storage. Git history, signed manifests,
published artifacts, and model qualification remain unchanged.
