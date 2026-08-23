export function resolvePostNormContract(normalization) {
  const postNormEps = normalization.postNormEps ?? normalization.rmsNormEps;
  const postNormWeightOffset = normalization.postNormWeightOffset
    ?? normalization.rmsNormWeightOffset;
  if (!Number.isFinite(postNormEps) || postNormEps <= 0) {
    throw new Error('normalization.postNormEps must be null or a positive finite number');
  }
  if (typeof postNormWeightOffset !== 'boolean') {
    throw new Error('normalization.postNormWeightOffset must be null or boolean');
  }
  return { postNormEps, postNormWeightOffset };
}

export function postNormContractMatchesBase(config) {
  return (config.postNormEps ?? config.rmsNormEps) === config.rmsNormEps
    && (config.postNormWeightOffset ?? config.rmsNormWeightOffset) === config.rmsNormWeightOffset;
}
