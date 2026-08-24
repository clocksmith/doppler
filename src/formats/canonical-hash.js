import { sha256BytesHex } from './sha256.js';

const HASH_NAMESPACES = new Set(['artifact', 'transcript', 'plan', 'integrity']);

function assertFiniteJsonNumber(value, path) {
  if (!Number.isFinite(value)) {
    throw new Error(`canonical hash: non-finite number at ${path}`);
  }
}

function canonicalizeValue(value, path = '$') {
  if (value === null) {
    return 'null';
  }
  if (value === undefined) {
    return undefined;
  }
  const valueType = typeof value;
  if (valueType === 'string' || valueType === 'boolean') {
    return JSON.stringify(value);
  }
  if (valueType === 'number') {
    assertFiniteJsonNumber(value, path);
    return JSON.stringify(value);
  }
  if (valueType === 'bigint' || valueType === 'function' || valueType === 'symbol') {
    throw new Error(`canonical hash: unsupported value at ${path}`);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item, index) => {
      const normalized = canonicalizeValue(item, `${path}[${index}]`);
      return normalized === undefined ? 'null' : normalized;
    }).join(',')}]`;
  }
  if (value instanceof Uint8Array) {
    return JSON.stringify(Array.from(value));
  }
  if (value instanceof ArrayBuffer) {
    return JSON.stringify(Array.from(new Uint8Array(value)));
  }
  if (valueType === 'object') {
    const entries = Object.entries(value)
      .filter(([, entryValue]) => entryValue !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    const rendered = entries.map(([key, entryValue]) => {
      const normalized = canonicalizeValue(entryValue, `${path}.${key}`);
      return `${JSON.stringify(key)}:${normalized}`;
    });
    return `{${rendered.join(',')}}`;
  }
  throw new Error(`canonical hash: unsupported value at ${path}`);
}

export function canonicalizeJson(value) {
  return canonicalizeValue(value);
}

export function hashBytesSha256(bytes) {
  const view = bytes instanceof Uint8Array
    ? bytes
    : (bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  return `sha256:${sha256BytesHex(view)}`;
}

export function computeCanonicalSha256(value) {
  return hashBytesSha256(new TextEncoder().encode(canonicalizeJson(value)));
}

export function computeNamespacedCanonicalSha256(namespace, value) {
  if (!HASH_NAMESPACES.has(namespace)) {
    throw new Error(`canonical hash: unsupported namespace "${namespace}"`);
  }
  return `${namespace}:${computeCanonicalSha256(value)}`;
}
