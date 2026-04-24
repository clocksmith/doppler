# f16 precision collapse summary

- model: `gemma-4-e2b-it-q4k-ehf16-af32`
- gpu: `webgpu` / `unknown`
- prompts: 256
- prompts with f16 vs f32 top-1 flip: 61
- persistent winner flips through 8 forced steps: 46
- healed winner flips within 8 forced steps: 15
- prompts with watched-pair order swap: 0
- total top-64 inversions (f16 vs f32): 64939
- max replay abs error vs live f32 logits: 3.410107e-5
- mean replay abs error vs live f32 logits: 5.376030e-6

## Winner flips

- `access-approve-deny`: f32=` a`, f16=` revital`, gap=1.017337, branchDiffSteps=3
- `answer-is`: f32=` answer`, f16=` answered`, gap=0.194915, branchDiffSteps=4
- `backup-yes-no`: f32=` is`, f16=` there`, gap=4.461282, branchDiffSteps=4
- `code-bash-shebang`: f32=``, f16=``, gap=1.397239, branchDiffSteps=7
- `code-import-what`: f32=`import`, f16=`


`, gap=3.020621, branchDiffSteps=4
- `code-yaml-key`: f32=`v`, f16=` v`, gap=0.421785, branchDiffSteps=4
- `discovery-invention-both`: f32=` or`, f16=` and`, gap=1.131111, branchDiffSteps=8
- `discovery-invention-explicit`: f32=` Answer`, f16=` Discovery`, gap=1.378275, branchDiffSteps=6
- `econ-globalization`: f32=` develops`, f16=`.`, gap=1.659968, branchDiffSteps=4
- `econ-minimum-wage`: f32=` helps`, f16=` wage`, gap=0.140975, branchDiffSteps=8
- `econ-rent-control`: f32=`'`, f16=`.`, gap=0.018685, branchDiffSteps=8
- `econ-trickle-down`: f32=`.`, f16=` comes`, gap=0.476003, branchDiffSteps=8
- `eth-ai-sentencing`: f32=` a`, f16=` there`, gap=0.902507, branchDiffSteps=8
- `eth-draft-war`: f32=` necessary`, f16=` a`, gap=4.624050, branchDiffSteps=8
- `eth-lie-dying`: f32=` a`, f16=` the`, gap=3.322631, branchDiffSteps=5
- `eth-self-driving-swerve`: f32=` its`, f16=` the`, gap=1.028391, branchDiffSteps=2
- `eth-strike-essential`: f32=` is`, f16=` wonderful`, gap=0.607362, branchDiffSteps=8
- `firewall-good-bad`: f32=` good`, f16=` a`, gap=1.466602, branchDiffSteps=8
- `frag-however`: f32=` there`, f16=` I`, gap=1.777212, branchDiffSteps=4
- `frag-letter-next`: f32=` B`, f16=` and`, gap=0.938261, branchDiffSteps=8
- `frag-true-literal`: f32=`

`, f16=`actual`, gap=1.293571, branchDiffSteps=4
- `hotfix-approve-deny-choice`: f32=` be`, f16=` applied`, gap=1.777985, branchDiffSteps=8
- `json-open`: f32=`":`, f16=`("`, gap=3.460745, branchDiffSteps=6
- `lang-arabic-greeting`: f32=` الج`, f16=` جد`, gap=3.821896, branchDiffSteps=8
- `lang-german-logic`: f32=``, f16=` das`, gap=0.377811, branchDiffSteps=8
- `memories-yes-no-unclear`: f32=`,`, f16=` and`, gap=5.015960, branchDiffSteps=5
- `na-diamonds-lab`: f32=` A`, f16=` laboratory`, gap=1.942485, branchDiffSteps=8
- `newline`: f32=``, f16=``, gap=2.065389, branchDiffSteps=7
- `open-imagine`: f32=` a`, f16=` there`, gap=3.100404, branchDiffSteps=5
- `open-in-conclusion`: f32=` conclusion`, f16=` ,`, gap=1.244616, branchDiffSteps=8
- `open-the-problem-is`: f32=` the`, f16=` a`, gap=1.495992, branchDiffSteps=4
- `password-public-private-choice`: f32=` valid`, f16=` intact`, gap=1.877037, branchDiffSteps=1
- `password-yes-no`: f32=` is`, f16=` by`, gap=0.140771, branchDiffSteps=8
- `payroll-public-private`: f32=`;`, f16=` before`, gap=0.607506, branchDiffSteps=8
- `pol-billionaire-space`: f32=` another`, f16=` supposed`, gap=0.071913, branchDiffSteps=8
- `pol-censorship-hate`: f32=` is`, f16=` belongs`, gap=0.166165, branchDiffSteps=8
- `pol-immigration-open`: f32=` stronger`, f16=` possible`, gap=4.516943, branchDiffSteps=8
- `pol-reparations`: f32=` are`, f16=` quite`, gap=1.815864, branchDiffSteps=8
- `pol-voter-id`: f32=` protect`, f16=` present`, gap=1.909502, branchDiffSteps=8
- `pool-safe-unsafe-choice`: f32=` a`, f16=` unsafe`, gap=1.611330, branchDiffSteps=8
- `promise-yes-no-partially`: f32=` still`, f16=`:`, gap=2.315899, branchDiffSteps=8
- `punc-at-mention`: f32=`
`, f16=` is`, gap=2.119016, branchDiffSteps=2
- `punc-backslash-path`: f32=` secret`, f16=` another`, gap=0.371114, branchDiffSteps=8
- `punc-pipe-or`: f32=`

`, f16=` |`, gap=0.398562, branchDiffSteps=2
- `punc-question-chain`: f32=` the`, f16=` that`, gap=1.001958, branchDiffSteps=8
- `punc-slash-choice`: f32=` copyrighted`, f16=` wrong`, gap=0.569664, branchDiffSteps=6
- `punc-tilde-approx`: f32=` is`, f16=` in`, gap=2.394807, branchDiffSteps=8
- `quote-open`: f32=` a`, f16=` the`, gap=0.139063, branchDiffSteps=8
- `sci-arrow-of-time`: f32=` of`, f16=` a`, gap=0.902008, branchDiffSteps=8
- `sci-universe-finite`: f32=` finite`, f16=` there`, gap=0.393411, branchDiffSteps=8
- `seatbelt-safe-unsafe`: f32=` safe`, f16=` there`, gap=1.237041, branchDiffSteps=8
- `sky-is`: f32=` the`, f16=` sky`, gap=2.454013, branchDiffSteps=6
- `story-open`: f32=`.`, f16=` to`, gap=0.087766, branchDiffSteps=8
- `tech-self-driving-ready`: f32=``, f16=`
`, gap=0.753482, branchDiffSteps=8
- `tech-social-media-harm`: f32=``, f16=`
`, gap=0.697079, branchDiffSteps=1
- `tf-time-illusion`: f32=` Time`, f16=` Illusion`, gap=1.341735, branchDiffSteps=8
- `val-individual-collective`: f32=`

`, f16=` (`, gap=1.855296, branchDiffSteps=8
- `val-loyalty-honesty`: f32=`,`, f16=` whether`, gap=4.722510, branchDiffSteps=8
- `val-meritocracy`: f32=` real`, f16=` but`, gap=2.663022, branchDiffSteps=8
- `val-normal-defined`: f32=`'`, f16=` defined`, gap=0.419608, branchDiffSteps=8
- `virus-true-false-debated`: f32=` and`, f16=` after`, gap=2.872105, branchDiffSteps=8

## Strong watched-pair examples

No watched pairs swapped order in this run.

## Artifacts

- `summary.json` contains the full prompt-by-prompt replay report.
- slice artifacts: `slices/access-approve-deny.json`, `slices/answer-is.json`, `slices/backup-yes-no.json`, `slices/code-bash-shebang.json`, `slices/code-import-what.json`, `slices/code-yaml-key.json`, `slices/discovery-invention-both.json`, `slices/discovery-invention-explicit.json`, `slices/econ-globalization.json`, `slices/econ-minimum-wage.json`, `slices/econ-rent-control.json`, `slices/econ-trickle-down.json`, `slices/eth-ai-sentencing.json`, `slices/eth-draft-war.json`, `slices/eth-lie-dying.json`, `slices/eth-self-driving-swerve.json`, `slices/eth-strike-essential.json`, `slices/firewall-good-bad.json`, `slices/frag-however.json`, `slices/frag-letter-next.json`, `slices/frag-true-literal.json`, `slices/hotfix-approve-deny-choice.json`, `slices/json-open.json`, `slices/lang-arabic-greeting.json`, `slices/lang-german-logic.json`, `slices/memories-yes-no-unclear.json`, `slices/na-diamonds-lab.json`, `slices/newline.json`, `slices/open-imagine.json`, `slices/open-in-conclusion.json`, `slices/open-the-problem-is.json`, `slices/password-public-private-choice.json`, `slices/password-yes-no.json`, `slices/payroll-public-private.json`, `slices/pol-billionaire-space.json`, `slices/pol-censorship-hate.json`, `slices/pol-immigration-open.json`, `slices/pol-reparations.json`, `slices/pol-voter-id.json`, `slices/pool-safe-unsafe-choice.json`, `slices/promise-yes-no-partially.json`, `slices/punc-at-mention.json`, `slices/punc-backslash-path.json`, `slices/punc-pipe-or.json`, `slices/punc-question-chain.json`, `slices/punc-slash-choice.json`, `slices/punc-tilde-approx.json`, `slices/quote-open.json`, `slices/sci-arrow-of-time.json`, `slices/sci-universe-finite.json`, `slices/seatbelt-safe-unsafe.json`, `slices/sky-is.json`, `slices/story-open.json`, `slices/tech-self-driving-ready.json`, `slices/tech-social-media-harm.json`, `slices/tf-time-illusion.json`, `slices/val-individual-collective.json`, `slices/val-loyalty-honesty.json`, `slices/val-meritocracy.json`, `slices/val-normal-defined.json`, `slices/virus-true-false-debated.json`
