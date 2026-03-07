import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { parseJsonl } from './datasets/jsonl.js';

function asTokenSequence(text) {
  return String(text ?? '')
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function extractCharacterNgrams(text, n) {
  const normalized = Array.from(String(text ?? '').trim());
  if (normalized.length < n) {
    return new Map();
  }
  const grams = new Map();
  for (let index = 0; index <= normalized.length - n; index += 1) {
    const gram = normalized.slice(index, index + n).join('');
    grams.set(gram, (grams.get(gram) || 0) + 1);
  }
  return grams;
}

function countOverlap(source, target) {
  let overlap = 0;
  for (const [key, sourceCount] of source.entries()) {
    const targetCount = target.get(key) || 0;
    overlap += Math.min(sourceCount, targetCount);
  }
  return overlap;
}

function computeBleuStats(hypotheses, references, maxOrder = 4) {
  const matchesByOrder = new Array(maxOrder).fill(0);
  const possibleByOrder = new Array(maxOrder).fill(0);
  let hypothesisLength = 0;
  let referenceLength = 0;

  for (let index = 0; index < hypotheses.length; index += 1) {
    const hypothesis = asTokenSequence(hypotheses[index]);
    const reference = asTokenSequence(references[index]);
    hypothesisLength += hypothesis.length;
    referenceLength += reference.length;
    for (let order = 1; order <= maxOrder; order += 1) {
      const hypothesisCounts = new Map();
      const referenceCounts = new Map();
      for (let tokenIndex = 0; tokenIndex <= hypothesis.length - order; tokenIndex += 1) {
        const ngram = hypothesis.slice(tokenIndex, tokenIndex + order).join('\u0001');
        hypothesisCounts.set(ngram, (hypothesisCounts.get(ngram) || 0) + 1);
      }
      for (let tokenIndex = 0; tokenIndex <= reference.length - order; tokenIndex += 1) {
        const ngram = reference.slice(tokenIndex, tokenIndex + order).join('\u0001');
        referenceCounts.set(ngram, (referenceCounts.get(ngram) || 0) + 1);
      }
      matchesByOrder[order - 1] += countOverlap(hypothesisCounts, referenceCounts);
      possibleByOrder[order - 1] += Math.max(0, hypothesis.length - order + 1);
    }
  }

  return {
    matchesByOrder,
    possibleByOrder,
    hypothesisLength,
    referenceLength,
  };
}

export function computeBleuScore(hypotheses, references, options = {}) {
  const maxOrder = Number.isInteger(options.maxOrder) && options.maxOrder > 0
    ? options.maxOrder
    : 4;
  if (!Array.isArray(hypotheses) || !Array.isArray(references) || hypotheses.length !== references.length) {
    throw new Error('computeBleuScore requires equally sized hypothesis and reference arrays.');
  }
  if (hypotheses.length === 0) {
    return {
      score: 0,
      brevityPenalty: 0,
      precisions: new Array(maxOrder).fill(0),
      hypothesisLength: 0,
      referenceLength: 0,
    };
  }

  const stats = computeBleuStats(hypotheses, references, maxOrder);
  const precisions = [];
  let precisionLogSum = 0;
  for (let order = 0; order < maxOrder; order += 1) {
    const matches = stats.matchesByOrder[order];
    const possible = stats.possibleByOrder[order];
    const precision = possible === 0
      ? 0
      : ((matches + 1) / (possible + 1));
    precisions.push(precision);
    precisionLogSum += Math.log(Math.max(precision, 1e-16));
  }
  const brevityPenalty = stats.hypothesisLength > stats.referenceLength
    ? 1
    : Math.exp(1 - (stats.referenceLength / Math.max(stats.hypothesisLength, 1)));
  const score = brevityPenalty * Math.exp(precisionLogSum / maxOrder);
  return {
    score,
    brevityPenalty,
    precisions,
    hypothesisLength: stats.hypothesisLength,
    referenceLength: stats.referenceLength,
  };
}

export function computeChrfScore(hypotheses, references, options = {}) {
  const maxOrder = Number.isInteger(options.maxOrder) && options.maxOrder > 0
    ? options.maxOrder
    : 6;
  const beta = Number.isFinite(options.beta) && options.beta > 0 ? options.beta : 2;
  if (!Array.isArray(hypotheses) || !Array.isArray(references) || hypotheses.length !== references.length) {
    throw new Error('computeChrfScore requires equally sized hypothesis and reference arrays.');
  }
  if (hypotheses.length === 0) {
    return {
      score: 0,
      precision: 0,
      recall: 0,
    };
  }

  let precisionSum = 0;
  let recallSum = 0;
  for (let order = 1; order <= maxOrder; order += 1) {
    let overlap = 0;
    let hypothesisTotal = 0;
    let referenceTotal = 0;
    for (let index = 0; index < hypotheses.length; index += 1) {
      const hypothesisCounts = extractCharacterNgrams(hypotheses[index], order);
      const referenceCounts = extractCharacterNgrams(references[index], order);
      overlap += countOverlap(hypothesisCounts, referenceCounts);
      for (const value of hypothesisCounts.values()) {
        hypothesisTotal += value;
      }
      for (const value of referenceCounts.values()) {
        referenceTotal += value;
      }
    }
    precisionSum += hypothesisTotal > 0 ? (overlap / hypothesisTotal) : 0;
    recallSum += referenceTotal > 0 ? (overlap / referenceTotal) : 0;
  }

  const precision = precisionSum / maxOrder;
  const recall = recallSum / maxOrder;
  const betaSquared = beta * beta;
  const score = (precision + recall) === 0
    ? 0
    : ((1 + betaSquared) * precision * recall) / ((betaSquared * precision) + recall);
  return { score, precision, recall };
}

export function computeExactMatch(hypotheses, references) {
  if (!Array.isArray(hypotheses) || !Array.isArray(references) || hypotheses.length !== references.length) {
    throw new Error('computeExactMatch requires equally sized hypothesis and reference arrays.');
  }
  if (hypotheses.length === 0) {
    return { score: 0, matches: 0, total: 0 };
  }
  let matches = 0;
  for (let index = 0; index < hypotheses.length; index += 1) {
    if (String(hypotheses[index] ?? '').trim() === String(references[index] ?? '').trim()) {
      matches += 1;
    }
  }
  return {
    score: matches / hypotheses.length,
    matches,
    total: hypotheses.length,
  };
}

export function computeAccuracy(labels, predictions) {
  return computeExactMatch(predictions, labels);
}

export function computeEvalMetrics(evalKind, hypotheses, references, options = {}) {
  const normalizedKind = String(evalKind || '').trim();
  if (normalizedKind === 'translation') {
    const bleu = computeBleuScore(hypotheses, references, options.bleu || {});
    const chrf = computeChrfScore(hypotheses, references, options.chrf || {});
    return {
      bleu,
      chrf,
      primaryMetric: 'bleu',
      primaryScore: bleu.score,
    };
  }
  if (normalizedKind === 'text_generation') {
    const exactMatch = computeExactMatch(hypotheses, references);
    return {
      exactMatch,
      primaryMetric: 'exact_match',
      primaryScore: exactMatch.score,
    };
  }
  if (normalizedKind === 'classification') {
    const accuracy = computeAccuracy(references, hypotheses);
    return {
      accuracy,
      primaryMetric: 'accuracy',
      primaryScore: accuracy.score,
    };
  }
  if (normalizedKind === 'retrieval' || normalizedKind === 'custom') {
    throw new Error(`Eval kind "${normalizedKind}" requires a custom evaluator and is not yet implemented.`);
  }
  throw new Error(`Unsupported eval kind "${normalizedKind}".`);
}

export async function loadEvalDataset(datasetPath) {
  const absolutePath = resolve(String(datasetPath));
  const raw = await readFile(absolutePath, 'utf8');
  const rows = absolutePath.endsWith('.json')
    ? JSON.parse(raw)
    : parseJsonl(raw);
  if (!Array.isArray(rows)) {
    throw new Error(`Eval dataset "${absolutePath}" must be a JSON array or JSONL file.`);
  }
  return {
    absolutePath,
    rows,
    raw,
  };
}
