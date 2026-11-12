// Pure argument, timestamp, hashing, redaction and JSON helpers shared by the
// engine compare tools (compare-embeddings.js, compare-rerankers.js).
// Workload validation and fairness gates stay in each tool.
import crypto from 'node:crypto';
import fs from 'node:fs/promises';

const BOOLEAN_FLAGS = Object.freeze(['help', 'h', 'json', 'save', 'all', 'browser-console']);

/**
 * Parses `--key value` flags. Boolean flags take no value; repeatable flags
 * collect every value into an array; any other flag keeps its last value.
 * @param {string[]} argv
 * @param {{ repeatable?: readonly string[] }} [options]
 * @returns {Record<string, any>}
 */
export function parseCompareArgs(argv, { repeatable = [] } = {}) {
  /** @type {Record<string, any>} */
  const flags = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '-h') {
      flags.h = true;
      continue;
    }
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (BOOLEAN_FLAGS.includes(key)) {
      flags[key] = true;
      continue;
    }
    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --${key}`);
    }
    if (repeatable.includes(key)) {
      if (!Array.isArray(flags[key])) flags[key] = [];
      flags[key].push(value);
    } else {
      flags[key] = value;
    }
    i += 1;
  }
  return flags;
}

/**
 * @template F
 * @param {unknown} value
 * @param {F} fallback
 * @param {string} label
 * @returns {number | F}
 */
export function parsePositiveInteger(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

/**
 * @template F
 * @param {unknown} value
 * @param {readonly string[]} allowed
 * @param {F} fallback
 * @param {string} label
 * @returns {string | F}
 */
export function parseChoice(value, allowed, fallback, label) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim();
  if (allowed.includes(normalized)) return normalized;
  const lower = normalized.toLowerCase();
  if (allowed.includes(lower)) return lower;
  throw new Error(`${label} must be one of: ${allowed.join(', ')}`);
}

/**
 * Returns an ISO-8601 timestamp from an ISO string or epoch milliseconds, or
 * the current time when no value is given.
 * @param {unknown} rawValue
 * @param {string} label
 * @returns {string}
 */
export function parseTimestampValue(rawValue, label) {
  if (rawValue == null || rawValue === '') return new Date().toISOString();
  const trimmed = String(rawValue).trim();
  const asMs = /^[-+]?\d+$/.test(trimmed) ? Number(trimmed) : NaN;
  const parsed = Number.isFinite(asMs) ? new Date(asMs) : new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`${label} must be ISO-8601 or epoch milliseconds`);
  }
  return parsed.toISOString();
}

/**
 * @param {string} timestamp
 * @returns {string} UTC `YYYYMMDDTHHMMSS`
 */
export function compactTimestamp(timestamp) {
  const d = new Date(timestamp);
  /** @param {number} n @param {number} [w] */
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

/** @param {string} text */
export function hashText(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

/**
 * Reads a JSON file and records the SHA-256 of its exact bytes.
 * @param {string} filePath
 * @returns {Promise<{ payload: any, source: string, sourceSha256: string }>}
 */
export async function readJsonWithHash(filePath) {
  const raw = await fs.readFile(filePath, 'utf-8');
  return {
    payload: JSON.parse(raw),
    source: filePath,
    sourceSha256: hashText(raw),
  };
}

/** @param {unknown} text */
export function redactSecrets(text) {
  return String(text ?? '')
    .replace(/hfToken=hf_[A-Za-z0-9._-]+/g, 'hfToken=<redacted>')
    .replace(/\bhf_[A-Za-z0-9._-]+\b/g, '<HF_TOKEN_REDACTED>')
    .replace(/(authorization:\s*bearer\s+)[^\s"'`]+/gi, '$1<redacted>');
}

/**
 * Parses command stdout as JSON, retrying from the first `{` when log lines
 * precede the payload.
 * @param {unknown} stdout
 * @param {string} label
 * @returns {any}
 */
export function parseJsonBlock(stdout, label) {
  const text = String(stdout ?? '').trim();
  if (!text) {
    throw new Error(`${label} produced no output`);
  }
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf('{');
    if (start >= 0) {
      try {
        return JSON.parse(text.slice(start));
      } catch { /* fall through */ }
    }
  }
  throw new Error(`${label} did not produce strict JSON. Output tail:\n${redactSecrets(text.slice(-2000))}`);
}
