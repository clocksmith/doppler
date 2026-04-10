function finiteNumberOrNull(value) {
  return Number.isFinite(value) ? value : null;
}

export function probToPerplexity(prob) {
  const safeProb = Math.max(1e-10, Math.min(1, Number.isFinite(prob) ? prob : 0));
  return 1 / safeProb;
}

export function quantile(values, ratio) {
  const sorted = Array.isArray(values)
    ? values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
    : [];
  if (sorted.length === 0) {
    return null;
  }
  if (sorted.length === 1) {
    return sorted[0];
  }
  const clampedRatio = Math.min(1, Math.max(0, ratio));
  const index = (sorted.length - 1) * clampedRatio;
  const lowerIndex = Math.floor(index);
  const upperIndex = Math.ceil(index);
  if (lowerIndex === upperIndex) {
    return sorted[lowerIndex];
  }
  const weight = index - lowerIndex;
  return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
}

export function summarizePerplexityRecords(records, options = {}) {
  const lowerQuantile = Number.isFinite(options.lowerQuantile) ? options.lowerQuantile : 0.05;
  const upperQuantile = Number.isFinite(options.upperQuantile) ? options.upperQuantile : 0.95;
  const perplexities = (Array.isArray(records) ? records : [])
    .map((record) => finiteNumberOrNull(record?.perplexity))
    .filter((value) => value != null);

  if (perplexities.length === 0) {
    return {
      count: 0,
      min: null,
      max: null,
      displayMin: null,
      displayMax: null,
      extremeLowCount: 0,
      extremeHighCount: 0,
    };
  }

  const min = Math.min(...perplexities);
  const max = Math.max(...perplexities);
  const displayMin = perplexities.length < 8 ? min : (quantile(perplexities, lowerQuantile) ?? min);
  const displayMax = perplexities.length < 8 ? max : (quantile(perplexities, upperQuantile) ?? max);

  let extremeLowCount = 0;
  let extremeHighCount = 0;
  for (const value of perplexities) {
    if (value < displayMin) {
      extremeLowCount += 1;
    } else if (value > displayMax) {
      extremeHighCount += 1;
    }
  }

  return {
    count: perplexities.length,
    min,
    max,
    displayMin,
    displayMax,
    extremeLowCount,
    extremeHighCount,
  };
}

export function normalizePerplexity(perplexity, minPerplexity, maxPerplexity) {
  const safePerplexity = Math.max(1, Number.isFinite(perplexity) ? perplexity : 1);
  const safeMin = Math.max(1, Number.isFinite(minPerplexity) ? minPerplexity : 1);
  const safeMax = Math.max(safeMin, Number.isFinite(maxPerplexity) ? maxPerplexity : safeMin);
  const logMin = Math.log(safeMin);
  const logMax = Math.log(safeMax);
  if (logMax <= logMin) {
    return 0;
  }
  const clamped = Math.min(safeMax, Math.max(safeMin, safePerplexity));
  return Math.min(1, Math.max(0, (Math.log(clamped) - logMin) / (logMax - logMin)));
}

export function enrichTopKCandidates(topK) {
  return (Array.isArray(topK) ? topK : []).map((candidate) => ({
    token: candidate.token,
    text: candidate.text ?? `[${candidate.token}]`,
    logit: finiteNumberOrNull(candidate.logit),
    probability: finiteNumberOrNull(candidate.prob),
    perplexity: probToPerplexity(candidate.prob),
  }));
}

export function buildTokenTraceRecord(record, index = 0) {
  const confidence = finiteNumberOrNull(record?.confidence) ?? 0;
  const topK = enrichTopKCandidates(record?.topK);
  const selectedRankIndex = topK.findIndex((candidate) => candidate.token === record?.tokenId);
  const selectedCandidate = selectedRankIndex >= 0 ? topK[selectedRankIndex] : null;
  return {
    index,
    tokenId: record?.tokenId ?? null,
    text: record?.text ?? '',
    confidence,
    perplexity: probToPerplexity(confidence),
    selectedRank: selectedRankIndex >= 0 ? selectedRankIndex + 1 : null,
    selectedLogit: selectedCandidate?.logit ?? null,
    topK,
  };
}
