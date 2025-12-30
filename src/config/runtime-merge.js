import { isPlainObject } from '../utils/plain-object.js';

// Runtime merge helper used by command runners and harness config resolution.
// Behavior:
// - undefined override: keep base value
// - null override: explicit disable/reset
// - plain objects: deep merge
// - other values: override replaces base
export function mergeRuntimeValues(base, override) {
  if (override === undefined) return base;
  if (override === null) return null;
  if (!isPlainObject(base) || !isPlainObject(override)) {
    return override;
  }
  const merged = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (value === undefined) continue;
    merged[key] = mergeRuntimeValues(base[key], value);
  }
  return merged;
}
