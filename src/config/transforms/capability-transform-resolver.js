import { TRANSFORMS } from './execution-graph-transforms.js';
import { matchesRule } from '../../gpu/kernels/rule-matcher.js';
import { loadJson } from '../../utils/load-json.js';

const rules = await loadJson(
  '../../rules/inference/capability-transforms.rules.json',
  import.meta.url,
  'Failed to load capability transform rules'
);

/**
 * Given device capabilities and the current execution graph context,
 * determine which transforms need to be applied.
 *
 * @param {Object} capabilities - { hasSubgroups, hasF16, hasSubgroupsF16, maxWorkgroupSize, maxBufferSize }
 * `hasSubgroupsF16` is a derived convenience bit (`hasSubgroups && hasF16`), not a separate WebGPU feature.
 * @param {Object} platform - { id, vendor, architecture }
 * @param {Object} graphContext - execution-v1 graph/dtype summary
 *   { activationDtype, kvDtype, modelId?, layerTypes?,
 *     hasDensePrefillProjectionKernel?,
 *     hasQ4DecodeProjectionKernel?, hasQ4PrefillProjectionKernel?,
 *     hasAvailableQ4PrefillProjectionKernel? }
 * @returns {{ transforms: Function[], names: string[], reason: string }}
 */
export function resolveCapabilityTransforms(capabilities, platform, graphContext) {
  const normalizedGraphContext = graphContext ?? {};
  const matchContext = {
    ...capabilities,
    ...normalizedGraphContext,
    activationDtype: normalizedGraphContext.activationDtype ?? null,
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
      };
    }
  }

  throw new Error(
    'CapabilityTransformResolver: no rule matched capabilities ' +
    JSON.stringify(matchContext)
  );
}

/**
 * Returns the safest alternate-plan widening transform for an f16 primary plan.
 *
 * Large-head models such as Gemma 4 E2B use decode attention geometry that the
 * generic full-f32 decode kernel cannot execute. On shader-f16 hardware, keep
 * KV on the f16 lane and widen activations only so the fallback plan stays
 * executable for headDim > 64.
 *
 * @param {Object} graphContext - { activationDtype, kvDtype, headDim?, modelId?, layerTypes? }
 * @returns {{ transform: Function, name: string, fallbackKvDtype: 'f16' | 'f32' } | null}
 */
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
