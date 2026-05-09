
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { parseJsonl } from './jsonl.js';

function concatTokens(tokens, maxLength) {
  if (!maxLength || tokens.length <= maxLength) {
    return tokens;
  }
  return tokens.slice(0, maxLength);
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
    const fullText = `${pair.prompt}${joinWith}${pair.completion}`;
    const tokens = tokenizer.encode(fullText);
    const clipped = concatTokens(tokens, maxLength);
    const { inputIds, targetIds } = buildCausalPair(clipped);
    if (inputIds.length > 0) {
      samples.push({
        id: pair.id,
        inputIds,
        targetIds,
        text: fullText,
        prompt: pair.prompt,
        completion: pair.completion,
      });
    }
  }
  return samples;
}
