import type { ModelIR } from '../config/model-ir.js';
import type { TargetPlan, TargetPlanKernelModule } from '../config/target-plan.js';
import type { DopplerPackV2, PackV2Artifact } from '../tooling/pack-v2.js';

export const FORGE_PIPELINE_VERSION = '1.0.0';

export interface StageInspectInput {
  modelDir: string;
  manifest?: Record<string, unknown> | null;
  config?: Record<string, unknown> | null;
}

export declare function stageInspect(input: StageInspectInput): Promise<{
  stage: 'inspect';
  ok: boolean;
  data: Record<string, unknown>;
}>;

export declare function stageAnalyze(intakeData: Record<string, unknown>): {
  stage: 'analyze';
  ok: boolean;
  modelIR: ModelIR;
  modelIRHash: `sha256:${string}`;
};

export declare function stageSpecialize(
  modelIR: ModelIR,
  kernelModules?: TargetPlanKernelModule[]
): {
  stage: 'specialize';
  ok: boolean;
  targetPlans: TargetPlan[];
  targetPlanHashes: `sha256:${string}`[];
};

export declare function stagePackage(params: {
  modelIR: ModelIR;
  targetPlans?: TargetPlan[];
  wgslModules?: TargetPlanKernelModule[];
  artifacts?: PackV2Artifact[];
  packId?: string | null;
}): {
  stage: 'package';
  ok: boolean;
  pack: DopplerPackV2;
  packId: string;
};
