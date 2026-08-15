import crypto from 'node:crypto';

export function canonicalizeJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalizeJson(entry)).join(',')}]`;
  }
  return `{${Object.keys(value).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalizeJson(value[key])}`
  )).join(',')}}`;
}

export function computeCanonicalJsonSha256(value) {
  const canonical = canonicalizeJson(value);
  return `sha256:${crypto.createHash('sha256').update(canonical).digest('hex')}`;
}
