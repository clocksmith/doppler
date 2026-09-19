// Application-owned verification of JSON receipts, independent of runtime internals.
import { createHash } from 'node:crypto';

export function computeCanonicalSha256(value) {
  const canonical = item => {
    if (Array.isArray(item)) return `[${item.map(canonical).join(',')}]`;
    if (item && typeof item === 'object') {
      return `{${Object.keys(item).sort((a, b) => a.localeCompare(b))
        .map(key => `${JSON.stringify(key)}:${canonical(item[key])}`).join(',')}}`;
    }
    if (item === undefined || (typeof item === 'number' && !Number.isFinite(item))) {
      throw new Error('Consumer evidence must be finite JSON.');
    }
    return JSON.stringify(item);
  };
  return `sha256:${createHash('sha256').update(canonical(value)).digest('hex')}`;
}
