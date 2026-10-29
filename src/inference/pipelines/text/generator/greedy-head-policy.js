import { isGreedySampling } from '../../../../rules/greedy-sampling.js';

/** @type {import('./greedy-head-policy.js').shouldUseGreedyLmHeadArgmaxFusion} */
export function shouldUseGreedyLmHeadArgmaxFusion(state, opts, samplingDefaults, repetitionPenalty) {
  const probes = state.runtimeConfig?.shared?.debug?.probes;
  return state.runtimeConfig?.inference?.session?.useGreedyLmHeadArgmaxFusion === true
    && (state.modelConfig?.logitOutputScale ?? 1) === 1
    && isGreedySampling(opts.temperature, opts.topK, samplingDefaults.greedyThreshold)
    && repetitionPenalty === 1
    && opts.presencePenalty === 0
    && (opts.suppressTokenIds?.length ?? 0) === 0
    && typeof opts.logitMaskFn !== 'function'
    && state.operatorDiagnostics == null
    && !(Array.isArray(probes) && probes.length > 0);
}
