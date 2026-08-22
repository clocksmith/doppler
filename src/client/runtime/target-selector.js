/**
 * Doppler TargetSelector: Selects the highest-priority prequalified TargetPlan
 * compatible with the active device capabilities.
 *
 * @module client/runtime/target-selector
 */

import { matchesDeviceCapability } from '../../config/target-plan.js';

/**
 * Selects the first compatible target plan from an array of pre-qualified target plans.
 *
 * @param {Array<object>} targetPlans Prequalified target plans ordered by preference (e.g. f16-subgroups > f16 > f32)
 * @param {object} deviceProfile Resolved WebGPU device features and limits
 * @returns {object} The selected TargetPlan
 * @throws {Error} If no prequalified target plan matches the device capabilities
 */
export function selectTargetPlan(targetPlans, deviceProfile) {
  if (!Array.isArray(targetPlans) || targetPlans.length === 0) {
    throw new Error('TargetSelector: Pack contains no target plans.');
  }
  if (!deviceProfile) {
    throw new Error('TargetSelector: deviceProfile is required for target selection.');
  }
  if (typeof deviceProfile.surface !== 'string' || !deviceProfile.surface.trim()) {
    throw new Error('TargetSelector: deviceProfile.surface is required for qualification selection.');
  }

  for (const plan of targetPlans) {
    const qualifiedForSurface = plan.qualification?.some((record) => (
      record.status === 'passed' && record.surface === deviceProfile.surface
    ));
    if (qualifiedForSurface && matchesDeviceCapability(plan, deviceProfile)) {
      return plan;
    }
  }

  const available = targetPlans.map((p) => p.targetId || 'unknown').join(', ');
  throw new Error(
    `TargetSelector: Device does not satisfy capability predicates and surface qualification for any prequalified target plan in Pack. Available targets: [${available}]. (surface: ${deviceProfile.surface}, hasF16: ${Boolean(deviceProfile.hasF16)}, hasSubgroups: ${Boolean(deviceProfile.hasSubgroups)})`
  );
}
