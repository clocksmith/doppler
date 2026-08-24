
import { matchesDeviceCapability } from '../../config/target-plan.js';

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
