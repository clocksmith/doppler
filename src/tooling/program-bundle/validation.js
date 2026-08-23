export function normalizeDigest(value, label) {
  const raw = String(value || '').trim().toLowerCase();
  const digest = raw.startsWith('sha256:') ? raw : `sha256:${raw}`;
  if (!/^sha256:[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`program bundle export: ${label} must be a sha256 digest.`);
  }
  return digest;
}

export function requirePlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`program bundle export: ${label} must be a non-null object.`);
  }
  return value;
}

export function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`program bundle export: ${label} must be a non-empty string.`);
  }
  return value.trim();
}

export function assertSha256ShardHashAlgorithm(hashAlgorithm, filename) {
  if (hashAlgorithm !== 'sha256') {
    throw new Error(
      `program bundle export: weight shard ${filename} requires manifest hashAlgorithm "sha256"; ` +
      `got "${hashAlgorithm ?? 'missing'}".`
    );
  }
}
