
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { parseJsonl } from './jsonl.js';

export const CAUSAL_LM_IGNORE_TARGET_ID = 0xffffffff;

function normalizeTokenIds(values, label) {
  const tokens = Array.from(values || []);
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (!Number.isInteger(token) || token < 0 || token > CAUSAL_LM_IGNORE_TARGET_ID) {
      throw new Error(`${label} token ${index + 1} must be a uint32 integer.`);
    }
  }
  return tokens;
}

function commonPrefixLength(left, right) {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

function truncatePromptTokens(tokens, budget) {
  if (tokens.length <= budget) {
    return tokens.slice();
  }
  const headCount = Math.ceil(budget / 2);
  const tailCount = budget - headCount;
  return [
    ...tokens.slice(0, headCount),
    ...(tailCount > 0 ? tokens.slice(tokens.length - tailCount) : []),
  ];
}

export function buildCausalPair(tokens) {
  if (tokens.length < 2) {
    return { inputIds: [], targetIds: [] };
  }
  return {
    inputIds: tokens.slice(0, tokens.length - 1),
    targetIds: tokens.slice(1),
  };
}

function readStringField(record, names, index, label) {
  for (const name of names) {
    if (Object.prototype.hasOwnProperty.call(record, name)) {
      const value = record[name];
      if (typeof value !== 'string') {
        throw new Error(`text pair row ${index + 1} field "${name}" for ${label} must be a string.`);
      }
      return { field: name, value };
    }
  }
  throw new Error(`text pair row ${index + 1} requires ${label}.`);
}

export function normalizeTextPair(record, index = 0) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    throw new Error(`text pair row ${index + 1} must be an object.`);
  }
  const prompt = readStringField(record, ['prompt', 'source', 'input'], index, 'prompt/source/input');
  const completion = readStringField(record, ['completion', 'target', 'output'], index, 'completion/target/output');
  return {
    id: String(record.id ?? record.rowId ?? `pair-${index + 1}`),
    prompt: prompt.value,
    completion: completion.value,
    promptField: prompt.field,
    completionField: completion.field,
  };
}

export function mapTextPairs(records) {
  const rows = Array.isArray(records) ? records : [];
  return rows.map((record, index) => normalizeTextPair(record, index));
}

export function parseTextPairsDataset(text, options = {}) {
  const sourceLabel = String(options.sourceLabel || 'dataset.jsonl');
  const parsed = sourceLabel.endsWith('.json')
    ? JSON.parse(String(text))
    : parseJsonl(String(text));
  if (!Array.isArray(parsed)) {
    throw new Error(`text pairs dataset "${sourceLabel}" must be a JSON array or JSONL records.`);
  }
  const rows = mapTextPairs(parsed);
  return {
    sourceLabel,
    rowCount: rows.length,
    rows,
  };
}

export async function loadTextPairsDataset(datasetPath, options = {}) {
  const source = String(datasetPath || '');
  if (!source) {
    throw new Error('loadTextPairsDataset requires a dataset path.');
  }
  const isUrl = /^https?:\/\//i.test(source);
  const text = isUrl
    ? await (options.fetch ? options.fetch(source) : fetch(source).then((res) => res.text()))
    : await (options.readFile ? options.readFile(resolve(source)) : readFile(resolve(source), 'utf8'));
  const parsed = parseTextPairsDataset(text, { sourceLabel: source });
  return {
    absolutePath: isUrl ? source : resolve(source),
    raw: text,
    ...parsed,
  };
}

export async function tokenizeTextPairs(tokenizer, pairs, options = {}) {
  if (!tokenizer || typeof tokenizer.encode !== 'function') {
    throw new Error('tokenizeTextPairs requires a tokenizer with encode()');
  }

  const {
    maxLength = null,
    joinWith = '',
  } = options;

  const samples = [];
  const normalizedPairs = mapTextPairs(pairs);
  for (const pair of normalizedPairs) {
    const promptText = `${pair.prompt}${joinWith}`;
    const fullText = `${promptText}${pair.completion}`;
    const promptTokens = normalizeTokenIds(
      tokenizer.encode(promptText),
      `text pair "${pair.id}" prompt`
    );
    const fullTokens = normalizeTokenIds(
      tokenizer.encode(fullText),
      `text pair "${pair.id}" full text`
    );
    const completionStart = commonPrefixLength(promptTokens, fullTokens);
    const promptPrefixTokens = fullTokens.slice(0, completionStart);
    const completionTokens = fullTokens.slice(completionStart);

    if (promptPrefixTokens.length < 1) {
      throw new Error(`text pair "${pair.id}" prompt produced no stable prefix tokens.`);
    }
    if (completionTokens.length < 1) {
      throw new Error(`text pair "${pair.id}" completion produced no supervised tokens.`);
    }

    let retainedPromptTokens = promptPrefixTokens;
    if (maxLength != null) {
      if (!Number.isInteger(maxLength) || maxLength < 2) {
        throw new Error('tokenizeTextPairs maxLength must be an integer >= 2 when provided.');
      }
      if (completionTokens.length >= maxLength) {
        throw new Error(
          `text pair "${pair.id}" completion requires ${completionTokens.length} tokens, `
          + `but maxLength ${maxLength} must also retain at least one prompt token.`
        );
      }
      retainedPromptTokens = truncatePromptTokens(
        promptPrefixTokens,
        maxLength - completionTokens.length
      );
    }

    const tokens = [...retainedPromptTokens, ...completionTokens];
    const { inputIds, targetIds } = buildCausalPair(tokens);
    const ignoredTargetCount = Math.max(0, retainedPromptTokens.length - 1);
    targetIds.fill(CAUSAL_LM_IGNORE_TARGET_ID, 0, ignoredTargetCount);
    const supervisedTokenCount = targetIds.length - ignoredTargetCount;
    if (inputIds.length > 0) {
      samples.push({
        id: pair.id,
        inputIds,
        targetIds,
        text: fullText,
        prompt: pair.prompt,
        completion: pair.completion,
        promptTokenCount: promptPrefixTokens.length,
        retainedPromptTokenCount: retainedPromptTokens.length,
        truncatedPromptTokenCount: promptPrefixTokens.length - retainedPromptTokens.length,
        completionTokenCount: completionTokens.length,
        ignoredTargetCount,
        supervisedTokenCount,
      });
    }
  }
  return samples;
}
