import { log, trace, isTraceEnabled } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';
import {
  applyRepetitionPenalty as applyScalarRepetitionPenalty,
  applyPresencePenalty as applyScalarPresencePenalty,
  sample as sampleTokens, softmax,
} from '../../token-sampling.js';

export { softmax };

export function applyRepetitionPenalty(logits, previousTokens, penalty, windowSize = getRuntimeConfig().inference.sampling.repetitionPenaltyWindow) {
  applyScalarRepetitionPenalty(logits, previousTokens, penalty, windowSize);
}

export function applyPresencePenalty(logits, previousTokens, penalty, windowSize = getRuntimeConfig().inference.sampling.repetitionPenaltyWindow) {
  applyScalarPresencePenalty(logits, previousTokens, penalty, windowSize);
}

export function sample(logits, opts) {
  return sampleTokens(logits, {
    ...opts,
    onCandidates: opts.debug ? candidates => {
      const top = candidates.slice(0, 5).map(candidate =>
        `"${opts.decode?.([candidate.token]) ?? '?'}"(${(candidate.prob * 100).toFixed(1)}%)`);
      trace.sample(`Top-5 (temp=${opts.temperature}, topK=${opts.topK}, topP=${opts.topP}): ${top.join(', ')}`);
    } : undefined,
  });
}

export function getTopK(logits, k = 5, decode) {
  const probs = softmax(new Float32Array(logits));


  const indexed = [];
  for (let i = 0; i < logits.length; i++) {
    indexed.push({ token: i, logit: logits[i], prob: probs[i] });
  }
  indexed.sort((a, b) => b.logit - a.logit);

  return indexed.slice(0, k).map(t => ({
    token: t.token,
    logit: t.logit,
    prob: t.prob,
    text: decode?.([t.token]) ?? `[${t.token}]`,
  }));
}


export function logitsSanity(logits, label, decode) {
  let min = Infinity;
  let max = -Infinity;
  let nanCount = 0;
  let infCount = 0;

  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (Number.isNaN(v)) {
      nanCount++;
    } else if (!Number.isFinite(v)) {
      infCount++;
    } else {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }

  const top5 = getTopK(logits, 5, decode);
  if (isTraceEnabled('sample')) {
    const top5Str = top5.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ');
    trace.sample(`${label} logits: min=${min.toFixed(2)}, max=${max.toFixed(2)} | top-5: ${top5Str}`);
  }

  if (nanCount > 0 || infCount > 0) {
    log.warn('Sampling', `${label} logits have ${nanCount} NaN, ${infCount} Inf values`);
  }

  return { min, max, nanCount, infCount, top5 };
}
