import type { TargetPlan, TargetPlanDeviceProfile } from '../../config/target-plan.js';

export type DeviceProfile = TargetPlanDeviceProfile;

export declare function selectTargetPlan(
  targetPlans: TargetPlan[],
  deviceProfile: DeviceProfile
): TargetPlan;
