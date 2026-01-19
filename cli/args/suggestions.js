


import { KNOWN_FLAGS } from './flags.js';

export function levenshteinDistance(a, b) {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  const prev = new Array(b.length + 1).fill(0);
  const curr = new Array(b.length + 1).fill(0);
  for (let j = 0; j <= b.length; j++) prev[j] = j;

  for (let i = 1; i <= a.length; i++) {
    curr[0] = i;
    const aChar = a.charCodeAt(i - 1);
    for (let j = 1; j <= b.length; j++) {
      const cost = aChar === b.charCodeAt(j - 1) ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1,
        curr[j - 1] + 1,
        prev[j - 1] + cost
      );
    }
    for (let j = 0; j <= b.length; j++) prev[j] = curr[j];
  }

  return prev[b.length];
}

export function normalizeFlag(flag) {
  return flag.replace(/^-+/, '').replace(/_/g, '-').toLowerCase();
}

export function suggestFlag(flag) {
  if (!flag.startsWith('--')) return null;
  if (!/[A-Z]/.test(flag)) return null;
  const kebab = `--${flag.slice(2).replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`)}`;
  return KNOWN_FLAGS.has(kebab) ? kebab : null;
}

export function resolveFlagAlias(flag) {
  if (!flag.startsWith('--')) return null;
  const raw = flag.slice(2);
  if (!raw) return null;
  const normalized = raw
    .replace(/_/g, '-')
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .toLowerCase();
  const candidate = `--${normalized}`;
  return KNOWN_FLAGS.has(candidate) ? candidate : null;
}

export function suggestClosestFlags(flag) {
  const camelSuggestion = suggestFlag(flag);
  if (camelSuggestion) return [camelSuggestion];

  const normalized = normalizeFlag(flag);
  const candidates = Array.from(KNOWN_FLAGS).filter((candidate) => {
    if (flag.startsWith('--')) return candidate.startsWith('--');
    return true;
  });
  const scored = candidates.map((candidate) => ({
    candidate,
    distance: levenshteinDistance(normalized, normalizeFlag(candidate)),
  }));

  scored.sort((a, b) => a.distance - b.distance || a.candidate.localeCompare(b.candidate));

  const best = scored[0];
  if (!best) return [];

  const maxDistance = Math.max(2, Math.floor(normalized.length / 3));
  return scored
    .filter((item) => item.distance <= maxDistance)
    .slice(0, 3)
    .map((item) => item.candidate);
}
