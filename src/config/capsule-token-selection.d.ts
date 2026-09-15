import type { TargetPlan } from './target-plan.js';

export interface CapsuleTokenSelection {
  schema: 'doppler.capsule-token-selection/v1';
  generationContract: 'doppler.generation-contract/v1';
  logitsDtype: 'f32';
  kernelModules: string[];
}
export declare const CAPSULE_TOKEN_SELECTION_CONTRACT: Readonly<{
  schema: string; declarationSchema: string; generationContract: string;
  logitsDtype: 'f32'; requiredKernels: readonly Readonly<{ operation: string; variant: string }>[];
}>;
export declare function validateCapsuleTokenSelection(plan: TargetPlan, modules?: {
  id: string; file: string; digest: string; sourceHash: string;
}[]): CapsuleTokenSelection;
