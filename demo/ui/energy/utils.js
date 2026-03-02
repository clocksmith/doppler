export function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function stableStringify(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableStringify(entry)).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).sort();
    const body = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(',');
    return `{${body}}`;
  }
  return JSON.stringify(value);
}

export function cloneSpec(spec) {
  if (typeof structuredClone === 'function') {
    return structuredClone(spec);
  }
  return JSON.parse(JSON.stringify(spec));
}

export function canonicalDepthForRound(round) {
  const idx = Number.parseInt(round, 10);
  if (idx === 0 || idx === 11) return 0;
  if (idx === 1 || idx === 12) return 1;
  if (idx === 2 || idx === 13) return 2;
  if (idx === 3 || idx === 14) return 3;
  return null;
}

export function requiredCachedNodes(maxDepth) {
  if (maxDepth >= 5) return 63;
  if (maxDepth >= 4) return 31;
  if (maxDepth >= 3) return 15;
  if (maxDepth >= 2) return 7;
  if (maxDepth >= 1) return 3;
  return 1;
}
