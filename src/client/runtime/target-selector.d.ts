import type { TargetPlan } from '../../config/target-plan.js';

export interface DeviceProfile {
  hasF16?: boolean;
  hasSubgroups?: boolean;
  maxBufferSize?: number;
  adapter?: {
    vendor?: string | null;
    architecture?: string | null;
    device?: string | null;
    description?: string | null;
  };
}

export declare function selectTargetPlan(
  targetPlans: TargetPlan[],
  deviceProfile: DeviceProfile
): TargetPlan;
