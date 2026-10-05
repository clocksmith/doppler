# Choice scoring

## Purpose

Obtain a typed choice from a model without generating an explanation. Applications
own questions, allowed actions and uncertainty policy. Doppler owns tokenization,
model computation, scoring and continuation cleanup.

## Import path

```js
import { openCapsule } from 'doppler-gpu/host';
import { validateChoiceScoringResult } from 'doppler-gpu/choice-scoring-contract';
```

## Audience

Applications that need bounded semantic judgments, including relevance checks.
This operation does not authorize tool execution, disclosure or side effects.

## Stability

New source capability; publication and model qualification remain separate.
Signed opening requires an explicit `scoreChoices` qualification for the selected
model, graph and host. Generation qualification does not qualify scoring. Current
Rig support uses causal-transformer ModelIR v1; heterogeneous ModelIR v2 scoring
has no implemented qualification path.

## Primary exports

`CHOICE_SCORING_CONTRACT`, `snapshotChoiceScoringRequest` and
`validateChoiceScoringResult` are pure public contracts. An opened model/session
provides `scoreChoices(request, { signal })`.

Prompts must already contain the selected model's formatting. Each answer label
must append exactly one distinct token **in that prompt's context**. Multi-token
labels, changed prompt tokenization, token aliases and oversized prompts reject
before inference. `maxSeqLen` is explicit. Results contain raw next-token logits,
the highest-scoring choice and `calibration: null`; logits are not correctness
probabilities. Ties select the first supplied choice.

## Minimal example

```js
const session = await openCapsule(capsuleUrl, {
  trustedSigners,
  acceptedTargetPlanDigests,
  requiredOperations: ['scoreChoices'],
});
try {
  const request = {
    prompt: formattedPrompt,
    choices: [{ id: 'relevant', label: 'yes' }, { id: 'irrelevant', label: 'no' }],
    maxSeqLen: 512,
  };
  const result = validateChoiceScoringResult(request,
    await session.scoreChoices(request, { signal }));
  console.log(result.selectedId, result.choices);
} finally {
  await session.close();
}
```

For receipt-backed delivery, use `session.executeOperation()` with operation
`{ name: 'scoreChoices', version: 1 }`, input `{ prompt, choices }`, options
`{ maxSeqLen }`, explicit assignment and byte/deadline limits. V1 and v2 return
one completed output without partial scores. Cancellation settles submitted work
before continuation reset; resident weights are retained for subsequent requests.

## Code pointers

- [Request/result contract](../../src/config/choice-scoring.js)
- [Numerical delegation](../../src/inference/choice-scoring.js)
- [Independent qualification contract](../../src/config/choice-scoring-reference.js)
- [Physical evidence and reproduction](../../reports/choice-scoring/README.md)

## Related surfaces

[Signed Run API](root.md), [Compatibility loading](compat.md),
[Incremental receipts](../capsule-streaming.md). Compatibility manifest loading
can screen a model; it does not establish signed-release qualification.
