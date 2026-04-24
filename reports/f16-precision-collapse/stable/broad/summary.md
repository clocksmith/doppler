# f16 precision collapse summary

- model: `gemma-3-270m-it-q4k-ehf16-af32`
- gpu: `webgpu` / `unknown`
- prompts: 256
- prompts with f16 vs f32 top-1 flip: 14
- persistent winner flips through 8 forced steps: 14
- healed winner flips within 8 forced steps: 0
- prompts with watched-pair order swap: 0
- total top-64 inversions (f16 vs f32): 39819
- max replay abs error vs live f32 logits: 6.103516e-5
- mean replay abs error vs live f32 logits: 4.108502e-6

## Winner flips

- `answer-is`: f32=`:`, f16=` `, gap=0.429276, branchDiffSteps=8
- `brakes-safe-unsafe-choice`: f32=` dangerous`, f16=` a`, gap=1.155497, branchDiffSteps=8
- `justice-revenge-explicit`: f32=`  `, f16=` `, gap=1.319517, branchDiffSteps=8
- `lang-swahili-stem`: f32=` k`, f16=` a`, gap=0.101159, branchDiffSteps=7
- `medical-public-private`: f32=` a`, f16=` clear`, gap=0.618055, branchDiffSteps=8
- `medical-public-private-choice`: f32=` confidential`, f16=` private`, gap=0.468728, branchDiffSteps=8
- `open-breaking-news`: f32=`  `, f16=` A`, gap=0.600500, branchDiffSteps=8
- `open-imagine`: f32=` the`, f16=` you`, gap=1.566261, branchDiffSteps=8
- `pol-abortion-access`: f32=` is`, f16=`'`, gap=0.639841, branchDiffSteps=8
- `pool-safe-unsafe-choice`: f32=` dangerous`, f16=` a`, gap=0.008898, branchDiffSteps=8
- `punc-backslash-path`: f32=`:`, f16=` the`, gap=0.025005, branchDiffSteps=8
- `sci-determinism`: f32=` not`, f16=` the`, gap=1.739901, branchDiffSteps=8
- `ssh-safe-unsafe-choice`: f32=` a`, f16=` not`, gap=0.661869, branchDiffSteps=8
- `su-mushroom`: f32=` unsafe`, f16=`
`, gap=2.399754, branchDiffSteps=8

## Strong watched-pair examples

No watched pairs swapped order in this run.

## Artifacts

- `summary.json` contains the full prompt-by-prompt replay report.
- slice artifacts: `slices/answer-is.json`, `slices/brakes-safe-unsafe-choice.json`, `slices/justice-revenge-explicit.json`, `slices/lang-swahili-stem.json`, `slices/medical-public-private.json`, `slices/medical-public-private-choice.json`, `slices/open-breaking-news.json`, `slices/open-imagine.json`, `slices/pol-abortion-access.json`, `slices/pool-safe-unsafe-choice.json`, `slices/punc-backslash-path.json`, `slices/sci-determinism.json`, `slices/ssh-safe-unsafe-choice.json`, `slices/su-mushroom.json`
