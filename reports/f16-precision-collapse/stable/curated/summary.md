# f16 precision collapse summary

- model: `gemma-3-270m-it-q4k-ehf16-af32`
- gpu: `webgpu` / `unknown`
- prompts: 12
- prompts with f16 vs f32 top-1 flip: 2
- persistent winner flips through 8 forced steps: 2
- healed winner flips within 8 forced steps: 0
- prompts with watched-pair order swap: 1
- total top-64 inversions (f16 vs f32): 1891
- max replay abs error vs live f32 logits: 2.098083e-5
- mean replay abs error vs live f32 logits: 3.984819e-6

## Winner flips

- `brakes-safe-unsafe-choice`: f32=` dangerous`, f16=` a`, gap=1.155497, branchDiffSteps=8
- `answer-is`: f32=`:`, f16=` `, gap=0.429276, branchDiffSteps=8

## Strong watched-pair examples

- `backup-yes-no-choice` ` yes` vs ` no`: f32= yes, f16= no, f32Gap=0.049947, f16Gap=-0.015625

## Artifacts

- `summary.json` contains the full prompt-by-prompt replay report.
- slice artifacts: `slices/red-traffic-go-stop.json`, `slices/earth-true-false.json`, `slices/sky-blue-green.json`, `slices/shareholders-society.json`, `slices/mask-fill.json`, `slices/mercy-cruelty.json`, `slices/backup-yes-no-choice.json`, `slices/brakes-safe-unsafe-choice.json`, `slices/answer-is.json`
