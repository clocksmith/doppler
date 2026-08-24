import { TRANSFORMS } from '../config/transforms/execution-graph-transforms.js';
import { matchesRule } from './rule-matcher.js';
import { loadJson } from '../utils/load-json.js';

const rules = await loadJson(
  './inference/capability-transforms.rules.json',
  import.meta.url,
  'Failed to load capability transform rules'
);

export function resolveCapabilityTransforms(capabilities, platform, graphContext) {
  const normalizedGraphContext = graphContext ?? {};
  const matchContext = {
    ...capabilities,
    ...normalizedGraphContext,
    requestedActivationDtype: normalizedGraphContext.requestedActivationDtype
      ?? normalizedGraphContext.activationDtype
      ?? null,
    activationDtype: normalizedGraphContext.activationDtype ?? null,
    mathDtype: normalizedGraphContext.mathDtype ?? null,
    accumDtype: normalizedGraphContext.accumDtype ?? null,
    kvDtype: normalizedGraphContext.kvDtype ?? null,
    modelId: normalizedGraphContext.modelId ?? 'unknown',
    platformId: platform?.id ?? 'unknown',
    platformVendor: platform?.vendor
      ?? platform?.detection?.vendor
      ?? capabilities?.adapterInfo?.vendor
      ?? 'unknown',
    platformArchitecture: platform?.architecture
      ?? platform?.detection?.architecture
      ?? capabilities?.adapterInfo?.architecture
      ?? 'unknown',
  };

  for (const rule of rules.capabilityTransforms) {
    if (matchesRule(rule.match, matchContext)) {
      const transforms = rule.transforms.map(name => {
        const fn = TRANSFORMS[name];
        if (!fn) {
          throw new Error(
            `CapabilityTransformResolver: unknown transform "${name}". ` +
            `Available: ${Object.keys(TRANSFORMS).join(', ')}`
          );
        }
        return fn;
      });
      return {
        transforms,
        names: rule.transforms,
        reason: rule.reason,
        kind: rule.kind,
        dtypeEffect: rule.dtypeEffect,
        evidence: Array.isArray(rule.evidence) ? [...rule.evidence] : [],
      };
    }
  }

  throw new Error(
    'CapabilityTransformResolver: no rule matched capabilities ' +
    JSON.stringify(matchContext)
  );
}

export function resolveFinitenessFallbackTransform(graphContext) {
  if (graphContext.activationDtype === 'f16') {
    if (Number.isFinite(graphContext.headDim) && graphContext.headDim > 64) {
      return {
        transform: TRANSFORMS.widenToF32Activations,
        name: 'widenToF32Activations',
        fallbackKvDtype: 'f16',
      };
    }
    return {
      transform: TRANSFORMS.widenToF32CorrectnessFallback,
      name: 'widenToF32CorrectnessFallback',
      fallbackKvDtype: 'f32',
    };
  }
  return null;
}
