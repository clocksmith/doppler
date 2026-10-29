import { validateGenerationField } from '../config/generation-contract.js';
import { selectRuleValue } from './rule-registry.js';

/** @type {import('./greedy-sampling.js').isGreedySampling} */
export function isGreedySampling(temperature, topK, greedyThreshold) {
  validateGenerationField('temperature', temperature);
  validateGenerationField('topK', topK);
  if (!Number.isFinite(greedyThreshold) || greedyThreshold < 0) {
    throw new Error('[Sampling] greedyThreshold must be a non-negative finite number.');
  }
  return selectRuleValue('kernels', 'sample', 'greedy', {
    temperature,
    topK,
    belowThreshold: temperature < greedyThreshold,
  });
}
