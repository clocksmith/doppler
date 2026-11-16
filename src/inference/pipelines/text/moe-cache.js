import { getRuntimeConfig } from '../../../config/runtime.js';
import { QK_K } from '../../../config/schema/index.js';

export function resolveMaxTokensPerExpert(numTokens, numExperts, topK, hiddenSize, activationDtype) {
  const routingConfig = getRuntimeConfig().inference.moe.routing;
  const {
    maxTokensPerExpert,
    maxTokensPerExpertHeadroom,
    maxTokensPerExpertMin,
    maxTokensPerExpertCap,
  } = routingConfig;
  for (const [name, value] of Object.entries({
    maxTokensPerExpert, maxTokensPerExpertHeadroom, maxTokensPerExpertMin, maxTokensPerExpertCap,
  })) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`runtime.inference.moe.routing.${name} must be a finite non-negative number, got ${String(value)}.`);
    }
  }

  let target = maxTokensPerExpert > 0
    ? maxTokensPerExpert
    : Math.ceil((numTokens * topK / Math.max(1, numExperts)) * maxTokensPerExpertHeadroom);

  target = Math.max(target, maxTokensPerExpertMin, 1);
  if (activationDtype === 'f16') {
    const bytesPerToken = hiddenSize * 2;
    const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
    const alignMultiple = QK_K / gcd(QK_K, bytesPerToken);
    let aligned = Math.ceil(target / alignMultiple) * alignMultiple;

    if (maxTokensPerExpertCap > 0) {
      const capAligned = Math.floor(maxTokensPerExpertCap / alignMultiple) * alignMultiple;
      aligned = Math.min(aligned, capAligned || alignMultiple);
    }
    return aligned;
  }

  if (maxTokensPerExpertCap > 0) {
    target = Math.min(target, maxTokensPerExpertCap);
  }
  return Math.min(target, numTokens);
}
