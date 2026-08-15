import { isPlainObject } from '../utils/plain-object.js';

const SCORE_POLICIES = new Set(['logit_difference', 'true_logit']);

function requireTokenId(value, label) {
  const tokenId = Number(value);
  if (!Number.isInteger(tokenId) || tokenId < 0) {
    throw new Error(`Manifest rerank config requires non-negative integer ${label}.`);
  }
  return tokenId;
}

function requireText(value, label, preserve = false) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`Manifest rerank config requires non-empty ${label}.`);
  }
  return preserve ? value : value.trim();
}

export function resolveRerankScoringConfig(pipeline) {
  const config = pipeline?.manifest?.inference?.rerank;
  if (!isPlainObject(config)) {
    throw new Error('Rerank workload requires manifest.inference.rerank scoring config.');
  }
  const format = requireText(config.format, 'format');
  if (format !== 'qwen3_yes_no_logit') {
    throw new Error(`Unsupported rerank scoring format "${format}".`);
  }
  const trueTokenId = requireTokenId(config.trueTokenId, 'trueTokenId');
  const falseTokenId = requireTokenId(config.falseTokenId, 'falseTokenId');
  if (trueTokenId === falseTokenId) {
    throw new Error('Manifest rerank config trueTokenId and falseTokenId must be distinct.');
  }
  const score = requireText(config.score, 'score');
  if (!SCORE_POLICIES.has(score)) {
    throw new Error(`Unsupported rerank score policy "${score}".`);
  }
  const probability = requireText(config.probability, 'probability');
  if (probability !== 'sigmoid') {
    throw new Error(`Unsupported rerank probability policy "${probability}".`);
  }
  return {
    format,
    instruction: requireText(config.instruction, 'instruction'),
    inputTemplate: requireText(config.inputTemplate, 'inputTemplate', true),
    prefix: requireText(config.prefix, 'prefix', true),
    suffix: requireText(config.suffix, 'suffix', true),
    trueToken: requireText(config.trueToken, 'trueToken'),
    trueTokenId,
    falseToken: requireText(config.falseToken, 'falseToken'),
    falseTokenId,
    score,
    probability,
  };
}

function replaceTemplate(template, values) {
  let output = template;
  for (const [key, value] of Object.entries(values)) {
    const placeholder = `{${key}}`;
    if (!output.includes(placeholder)) {
      throw new Error(`Manifest rerank inputTemplate is missing ${placeholder}.`);
    }
    output = output.split(placeholder).join(value);
  }
  return output;
}

export function formatRerankPrompt(query, document, scoringConfig) {
  const instruction = requireText(scoringConfig?.instruction, 'instruction');
  const input = replaceTemplate(
    requireText(scoringConfig?.inputTemplate, 'inputTemplate', true),
    {
      instruction,
      query: requireText(query, 'query'),
      document: requireText(document, 'document'),
    }
  );
  return `${requireText(scoringConfig?.prefix, 'prefix', true)}${input}${requireText(scoringConfig?.suffix, 'suffix', true)}`;
}

function computeScore(scoringConfig, trueLogit, falseLogit) {
  if (scoringConfig.score === 'logit_difference') return trueLogit - falseLogit;
  if (scoringConfig.score === 'true_logit') return trueLogit;
  throw new Error(`Unsupported rerank score policy "${scoringConfig.score}".`);
}

export function assertRerankLogitsVector(value) {
  if (!ArrayBuffer.isView(value) && !Array.isArray(value)) {
    throw new Error('Rerank prefillWithLogits result must include a logits vector.');
  }
  return value;
}

export function buildRerankScoreRecord(
  query,
  document,
  prompt,
  tokenCount,
  trueLogit,
  falseLogit,
  config,
  scoringPath,
  phase
) {
  if (!Number.isFinite(trueLogit) || !Number.isFinite(falseLogit)) {
    throw new Error(
      `Rerank logits missing finite yes/no scores at token IDs ${config.trueTokenId}/${config.falseTokenId}.`
    );
  }
  const score = computeScore(config, trueLogit, falseLogit);
  return {
    query,
    document,
    prompt,
    tokenCount,
    score,
    probability: 1 / (1 + Math.exp(-score)),
    trueLogit,
    falseLogit,
    trueTokenId: config.trueTokenId,
    falseTokenId: config.falseTokenId,
    scoringPath,
    phase,
  };
}

export async function scoreRerankDocument(
  pipeline,
  query,
  document,
  scoringConfig = null,
  options = {}
) {
  if (!pipeline || (
    typeof pipeline.prefillWithTokenLogits !== 'function'
    && typeof pipeline.prefillWithLogits !== 'function'
  )) {
    throw new Error(
      'Rerank workload requires pipeline.prefillWithTokenLogits() or pipeline.prefillWithLogits().'
    );
  }
  const config = scoringConfig ?? resolveRerankScoringConfig(pipeline);
  const prompt = formatRerankPrompt(query, document, config);
  pipeline.reset?.();
  const startedAt = performance.now();
  const selected = typeof pipeline.prefillWithTokenLogits === 'function';
  const callStartedAt = performance.now();
  const result = selected
    ? await pipeline.prefillWithTokenLogits(
      prompt,
      [config.trueTokenId, config.falseTokenId],
      { useChatTemplate: false, benchmark: options.benchmark === true }
    )
    : await pipeline.prefillWithLogits(prompt, {
      useChatTemplate: false,
      benchmark: options.benchmark === true,
    });
  const logits = assertRerankLogitsVector(result?.logits);
  const trueLogit = Number(logits[selected ? 0 : config.trueTokenId]);
  const falseLogit = Number(logits[selected ? 1 : config.falseTokenId]);
  return buildRerankScoreRecord(
    query,
    document,
    prompt,
    Number.isFinite(result?.tokens?.length) ? result.tokens.length : 0,
    trueLogit,
    falseLogit,
    config,
    selected ? 'selected-token-logits' : 'full-logits',
    {
      ...(isPlainObject(result?.phase) ? result.phase : {}),
      prefillCallMs: performance.now() - callStartedAt,
      totalMs: performance.now() - startedAt,
      promptChars: prompt.length,
    }
  );
}
