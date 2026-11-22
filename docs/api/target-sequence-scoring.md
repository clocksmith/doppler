# Target sequence scoring

## Purpose

Measure how likely a model is to produce an exact target string after a prompt:
the total natural-log probability `log P(target | prompt)` under teacher forcing.
Reverse-prompting games use it as a closeness score for prompts whose greedy output
misses the target, and to break ties. Applications own the game rules; Doppler
owns prompt formatting (for `messages`), tokenization, model computation and
continuation cleanup.

## Import path

```js
import { load } from 'dopplerx/compat';
import { validateTargetSequenceScoreResult } from 'dopplerx/host';
```

## Audience

Applications that compare prompts against a fixed target on a pinned model and
provider. The score describes the raw model distribution; it does not include
repetition penalties, logit masks or other sampling transforms an application may
apply while generating.

## Stability

New source capability on the compatibility model handle (`model.scoreTargetSequence`)
and on the scoped session returned by `runtime.open()` (`doppler.scoped-session/v1`),
which reports it as `supports('targetScoring')` and passes the handle's result through
unchanged.
Signed Capsule sessions do not expose it yet: like `scoreChoices`, it would need its
own operation qualification. It runs on the WebGPU text pipeline for the same
models that support `prefillWithTokenLogits`; models with hyper-connections or
replay-prefill decode (`decodeStrategy: "replay_prefill"`) fail fast. Physical
evidence so far: Gemma 3 270M (q4k) on Apple M3 / Metal through Node WebGPU.

## Primary exports

`TARGET_SEQUENCE_SCORING_CONTRACT`, `snapshotTargetSequenceScoringRequest` and
`validateTargetSequenceScoreResult` are pure public contracts. A loaded model
provides `scoreTargetSequence(request, { signal })`.

The request has exactly one of `prompt` (already formatted model text, used as is)
or `messages` (`system`/`user` turns, formatted with the model's chat template in
non-thinking mode and ending at the assistant turn), plus `target` and an explicit
`maxSeqLen`. The target is scored exactly as given, never trimmed.

The target tokens are the canonical tokenization of `prompt + target` after the
prompt's own tokens. If appending the target changes the prompt's tokens (for
example a leading newline merging with the prompt's final newline) or adds no
tokens, scoring rejects before inference. Targets longer than
`maxTargetTokens` (32) or sequences longer than `maxSeqLen` reject too.

One prefill runs over the prompt plus every target token except the last; the LM
head projects only the trailing rows that predict target tokens. Each
`tokenLogProbs[i]` is the log-softmax of the target token over the complete
vocabulary row (max-subtracted, float64 accumulation in index order).
`totalLogProb` is their index-order sum, `meanLogProb` divides by `tokenCount`,
and `roundedTotalLogProb` rounds the total to 0.01 (half toward +Infinity).
Scoring always starts from a fresh sequence: KV, linear-attention recurrent and
convolution state left by an earlier generation or score is cleared before the
prefill, so a score never depends on what ran before it. Generation state is
reset afterwards on success, failure and cancellation; resident weights stay
loaded.

**Reproducibility.** Values are reproducible for the same request on the same
Doppler version, model artifact, runtime profile and GPU/driver. They are not
promised identical across GPUs, browsers or WebGPU providers: kernels, reduction
order and f16 paths differ. Compare `roundedTotalLogProb` only between scores
computed on one pinned provider. Teacher-forced prefill numerics are not
bit-identical to token-by-token decode, so a score is not a proof that greedy
decoding produces the target; check the generated text for exact wins.

## Minimal example

```js
const model = await load({ url: modelUrl });
const request = {
  messages: [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: playerPrompt },
  ],
  target: 'Paris.',
  maxSeqLen: 512,
};
const result = validateTargetSequenceScoreResult(request,
  await model.scoreTargetSequence(request, { signal }));
console.log(result.roundedTotalLogProb, result.tokenLogProbs);
```

## Code pointers

- [Request/result contract](../../src/config/target-sequence-scoring.js)
- [Tokenization boundary and reset](../../src/inference/target-sequence-scoring.js)
- [Teacher-forced prefill](../../src/inference/pipelines/text/generator/adapters/target-scoring.js)
- [Log-softmax](../../src/inference/pipelines/text/logits/selected-log-softmax.js)

## Related surfaces

[Choice scoring](choice-scoring.md) scores single-token labels from raw logits;
[Compatibility loading](compat.md).
