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
 * @param {Object} platform - { id, vendor, architecture }
 * @param {Object} graphContext - { activationDtype, kvDtype }
 * @returns {{ transforms: Function[], names: string[], reason: string }}
 */
export function resolveCapabilityTransforms(capabilities, platform, graphContext) {
  for (const rule of rules.capabilityTransforms) {
    if (matchesRule(rule.match, capabilities)) {
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
    JSON.stringify(capabilities)
  );
}

/**
 * Returns widenToF32Activations when current activationDtype is f16,
 * or null when already f32 (no fallback available).
 *
 * @param {Object} graphContext - { activationDtype, kvDtype }
 * @returns {{ transform: Function, name: string } | null}
 */
export function resolveFinitenessFallbackTransform(graphContext) {
  if (graphContext.activationDtype === 'f16') {
    return {
      transform: TRANSFORMS.widenToF32Activations,
      name: 'widenToF32Activations',
    };
  }
  return null;
}
