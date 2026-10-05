import contract from './choice-scoring.json' with { type: 'json' };
import { freezeCapsuleV2 } from './capsule-v2.js';

export const CHOICE_SCORING_CONTRACT = freezeCapsuleV2(contract);

/** @param {unknown} input
 * @returns {Readonly<import('./choice-scoring.js').ChoiceScoringRequest>} */
export function snapshotChoiceScoringRequest(input) {
  const request = /** @type {Record<string, unknown> | null} */ (input);
  if (!request || typeof request !== 'object' || Array.isArray(request)
    || Object.keys(request).some(key => !['prompt', 'choices', 'maxSeqLen'].includes(key))) {
    throw new Error('Choice scoring requires prompt, choices and maxSeqLen.');
  }
  if (typeof request.prompt !== 'string' || !request.prompt.trim()) {
    throw new Error('Choice scoring requires a non-empty, already formatted prompt.');
  }
  if (typeof request.maxSeqLen !== 'number' || !Number.isSafeInteger(request.maxSeqLen) || request.maxSeqLen < 1) {
    throw new Error('Choice scoring requires a positive maxSeqLen.');
  }
  if (!Array.isArray(request.choices) || request.choices.length < contract.minChoices
    || request.choices.length > contract.maxChoices) {
    throw new Error(`Choice scoring requires ${contract.minChoices}–${contract.maxChoices} choices.`);
  }
  const ids = new Set();
  const labels = new Set();
  const choices = request.choices.map(value => {
    const choice = /** @type {Record<string, unknown> | null} */ (value);
    if (!choice || typeof choice !== 'object' || Array.isArray(choice)
      || Object.keys(choice).some(key => !['id', 'label'].includes(key))
      || typeof choice.id !== 'string' || !choice.id.trim()
      || typeof choice.label !== 'string' || !choice.label.trim()
      || ids.has(choice.id) || labels.has(choice.label)) {
      throw new Error('Choice scoring requires distinct, non-empty choice ids and labels.');
    }
    ids.add(choice.id);
    labels.add(choice.label);
    return { id: choice.id, label: choice.label };
  });
  return freezeCapsuleV2({ prompt: request.prompt, choices, maxSeqLen: request.maxSeqLen });
}

/** @param {import('./choice-scoring.js').ChoiceScoringRequest} input
 * @param {unknown} value
 * @returns {import('./choice-scoring.js').ChoiceScoringResult} */
export function validateChoiceScoringResult(input, value) {
  const result = /** @type {import('./choice-scoring.js').ChoiceScoringResult | null} */ (value);
  const request = snapshotChoiceScoringRequest(input);
  if (!result || result.schema !== contract.resultSchema || result.interpretation !== contract.interpretation
    || result.calibration !== contract.calibration || !Number.isSafeInteger(result.promptTokenCount)
    || result.promptTokenCount < 1 || result.promptTokenCount > request.maxSeqLen
    || !Array.isArray(result.choices) || result.choices.length !== request.choices.length) {
    throw new Error('Invalid choice scoring result contract.');
  }
  const tokens = new Set();
  /** @type {import('./choice-scoring.js').ChoiceScoringResult['choices'][number] | undefined} */
  let selected;
  for (const [index, choice] of result.choices.entries()) {
    if (choice?.id !== request.choices[index].id || choice?.label !== request.choices[index].label
      || !Number.isSafeInteger(choice.tokenId) || choice.tokenId < 0 || tokens.has(choice.tokenId)
      || !Number.isFinite(choice.logit)) throw new Error('Choice scoring result does not bind the requested labels.');
    tokens.add(choice.tokenId);
    if (!selected || choice.logit > selected.logit) selected = choice;
  }
  if (result.selectedId !== selected?.id) throw new Error('Choice scoring selection does not match its logits.');
  return result;
}
