import sourcePackageRegistry from '../config/source-packages/registry.json' with { type: 'json' };
import gemma4E2BPackageProfile from '../config/source-packages/litert/gemma-4-e2b-it.json' with { type: 'json' };
import gemma4E4BPackageProfile from '../config/source-packages/litert/gemma-4-e4b-it.json' with { type: 'json' };
import gemma412BPackageProfile from '../config/source-packages/litert/gemma-4-12b-it.json' with { type: 'json' };
import { cloneJsonValue } from '../utils/clone-json.js';

const PROFILE_MAP = new Map([
  ['litert/gemma-4-e2b-it', gemma4E2BPackageProfile],
  ['litert/gemma-4-e4b-it', gemma4E4BPackageProfile],
  ['litert/gemma-4-12b-it', gemma412BPackageProfile],
]);

function normalizeText(value) {
  return String(value || '').trim();
}

function normalizeLower(value) {
  return normalizeText(value).toLowerCase();
}

function resolveExecutionTemplateProfile(profileId) {
  const templateProfile = PROFILE_MAP.get(normalizeText(profileId));
  if (!templateProfile) {
    throw new Error(
      `direct-source package profile references unknown executionTemplateProfileId "${profileId}".`
    );
  }
  return templateProfile;
}

function resolveProfileLayerTypes(profile) {
  const manifestInference = profile?.runtime?.manifestInference ?? null;
  const explicitLayerTypes = manifestInference?.layerPattern?.layerTypes;
  if (Array.isArray(explicitLayerTypes) && explicitLayerTypes.length > 0) {
    return explicitLayerTypes.map((value) => normalizeText(value));
  }

  const layerPattern = manifestInference?.layerPattern ?? null;
  const numLayers = Number(profile?.runtime?.architecture?.numLayers ?? 0);
  const period = Number(layerPattern?.period ?? 0);
  const offset = Number(layerPattern?.offset ?? 0);
  if (
    layerPattern?.type !== 'every_n'
    || !Number.isInteger(numLayers)
    || numLayers <= 0
    || !Number.isInteger(period)
    || period <= 0
    || !Number.isInteger(offset)
  ) {
    throw new Error(
      `direct-source package profile "${profile?.id || '(unknown)'}" needs explicit layerPattern.layerTypes ` +
      'or a valid every_n layerPattern before an execution template can be retargeted.'
    );
  }

  const normalizedOffset = ((offset % period) + period) % period;
  const layerTypes = [];
  for (let layerIndex = 0; layerIndex < numLayers; layerIndex += 1) {
    const isGlobal = (((layerIndex - normalizedOffset) % period) + period) % period === 0;
    layerTypes.push(isGlobal ? 'full_attention' : 'sliding_attention');
  }
  return layerTypes;
}

function retargetExecutionTemplateLayers(execution, layerTypes) {
  const retargeted = cloneJsonValue(execution);
  const localLayers = [];
  const globalLayers = [];
  layerTypes.forEach((layerType, layerIndex) => {
    if (normalizeText(layerType) === 'full_attention') {
      globalLayers.push(layerIndex);
    } else {
      localLayers.push(layerIndex);
    }
  });

  if (!Array.isArray(retargeted?.prefill)) {
    return retargeted;
  }

  for (const group of retargeted.prefill) {
    if (!group || typeof group !== 'object' || !Array.isArray(group.layers)) {
      continue;
    }
    const steps = Array.isArray(group.steps) ? group.steps : [];
    const stepNames = steps
      .flatMap((step) => Array.isArray(step) ? step : [])
      .map((value) => normalizeText(value).toLowerCase())
      .filter(Boolean);
    const isGlobalGroup = stepNames.some((value) => value.includes('stream') || value.includes('global'));
    const isLocalGroup = stepNames.some((value) => value.includes('small') || value.includes('sliding'));
    if (isGlobalGroup) {
      group.layers = [...globalLayers];
    } else if (isLocalGroup) {
      group.layers = [...localLayers];
    }
  }

  return retargeted;
}

function materializePackageProfile(profile) {
  const execution = profile?.runtime?.manifestInference?.execution ?? null;
  if (execution && typeof execution === 'object') {
    return profile;
  }

  const templateProfileId = normalizeText(profile?.package?.litertlm?.executionTemplateProfileId);
  if (!templateProfileId) {
    return profile;
  }
  const templateProfile = resolveExecutionTemplateProfile(templateProfileId);
  const templateExecution = templateProfile?.runtime?.manifestInference?.execution ?? null;
  if (!templateExecution || typeof templateExecution !== 'object') {
    throw new Error(
      `direct-source package profile "${profile.id}" references executionTemplateProfileId "${templateProfileId}" ` +
      'but that template profile has no runtime.manifestInference.execution object.'
    );
  }
  const layerTypes = resolveProfileLayerTypes(profile);
  profile.runtime.manifestInference.execution = retargetExecutionTemplateLayers(templateExecution, layerTypes);
  return profile;
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
    return materializePackageProfile(cloneJsonValue(profile));
  }

  return null;
}
