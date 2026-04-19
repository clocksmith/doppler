import sourcePackageRegistry from '../config/source-packages/registry.json' with { type: 'json' };
import gemma4E2BPackageProfile from '../config/source-packages/litert/gemma-4-e2b-it.json' with { type: 'json' };
import { cloneJsonValue } from '../utils/clone-json.js';

const PROFILE_MAP = new Map([
  ['litert/gemma-4-e2b-it', gemma4E2BPackageProfile],
]);

function normalizeText(value) {
  return String(value || '').trim();
}

function normalizeLower(value) {
  return normalizeText(value).toLowerCase();
}

export function resolveDirectSourcePackageProfile(options = {}) {
  const sourceKind = normalizeLower(options.sourceKind);
  const packageBasename = normalizeLower(options.packageBasename);
  if (!sourceKind || !packageBasename) {
    return null;
  }

  const profiles = Array.isArray(sourcePackageRegistry?.profiles)
    ? sourcePackageRegistry.profiles
    : [];
  for (const entry of profiles) {
    const entrySourceKinds = Array.isArray(entry?.sourceKinds)
      ? entry.sourceKinds.map((value) => normalizeLower(value)).filter(Boolean)
      : [];
    if (!entrySourceKinds.includes(sourceKind)) {
      continue;
    }
    const packageBasenames = Array.isArray(entry?.packageBasenames)
      ? entry.packageBasenames.map((value) => normalizeLower(value)).filter(Boolean)
      : [];
    if (!packageBasenames.includes(packageBasename)) {
      continue;
    }
    const id = normalizeText(entry?.id);
    const profile = id ? PROFILE_MAP.get(id) : null;
    if (!profile) {
      throw new Error(
        `direct-source package profile registry is missing a loaded profile for "${id || '(empty)'}".`
      );
    }
    return cloneJsonValue(profile);
  }

  return null;
}
