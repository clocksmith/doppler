import { isPlainObject } from '../formats/plain-object.js';
import { log } from '../debug/index.js';
import { canonicalizeLegacyEnumValues } from './legacy-enum-values.js';

// Runtime merge helper used by command runners and harness config resolution.
// Behavior:
// - undefined override: keep base value
// - null override: explicit disable/reset
// - plain objects: deep merge
// - other values: override replaces base
// - legacy snake_case enum values in the override are read as kebab-case
export function mergeRuntimeValues(base, override, _path = '') {
  if (override === undefined) return base;
  if (_path === '') override = canonicalizeLegacyEnumValues(override);
  if (override === null) return null;
  if (!isPlainObject(base) || !isPlainObject(override)) {
    // Warn when an object is being replaced by a non-object or vice-versa,
    // which usually indicates a schema mismatch between base and override.
    if (isPlainObject(base) && !isPlainObject(override)) {
      log.warn('Config', `Type mismatch during merge at "${_path}": base is object, override is ${typeof override}`);
    } else if (!isPlainObject(base) && isPlainObject(override) && base != null) {
      log.warn('Config', `Type mismatch during merge at "${_path}": base is ${typeof base}, override is object`);
    }
    return override;
  }
  const merged = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (value === undefined) continue;
    const childPath = _path ? `${_path}.${key}` : key;
    merged[key] = mergeRuntimeValues(base[key], value, childPath);
  }
  return merged;
}
