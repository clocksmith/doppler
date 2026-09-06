import type { TargetPlan, TargetPlanDeviceProfile, TargetPlanSelectionPolicy } from '../../config/target-plan.js';

export type DeviceProfile = TargetPlanDeviceProfile;

export declare function selectTargetPlan(
  targetPlans: TargetPlan[],
  deviceProfile: DeviceProfile,
  selectionPolicy?: TargetPlanSelectionPolicy
): TargetPlan;
