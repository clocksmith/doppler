import type { OnboardingInput } from './model-onboarding.js';
export type OnboardingStage = 'conversion' | 'source-reference' | 'model-qualification' | 'capsule-construction' | 'capsule-qualification';
export interface OnboardingExecutionConfig {
  schema: 'doppler.model-onboarding-execution/v1'; operation: 'embed'; stages: OnboardingStage[];
  inputs: Record<'sourcePolicy' | 'frozenReference' | 'qualificationConfig' | 'capsuleConfig' | 'driver'
    | 'installedPackageReceipt' | 'installedPackageArchive' | 'license' | 'application', OnboardingInput>;
  sourceFiles: Array<OnboardingInput & { sizeBytes: number }>;
}
export interface OnboardingStageReceipt {
  schema: 'doppler.onboarding-stage-result/v1'; stage: OnboardingStage; passed: boolean;
  source: { repository: string; revision: string }; physicalExecution: boolean; data: Record<string, unknown>;
}
export interface OnboardingStageContext {
  stage: OnboardingStage; sourceRoot: string; outputDir: string; sourceIdentity: Record<string, unknown>; conversionPath: string;
  inputs: Record<string, string>; completed: Partial<Record<OnboardingStage, OnboardingStageReceipt>>;
}
export interface OnboardingExecutionPorts {
  runStage(context: OnboardingStageContext): Promise<string>;
  verifyStage(context: OnboardingStageContext, receipt: OnboardingStageReceipt): Promise<boolean>;
}
export interface OnboardingExecutionResult {
  schema: 'doppler.model-onboarding-execution-result/v1'; inputDigest: string; sourceIdentity: Record<string, unknown>;
  stages: Array<{ stage: OnboardingStage; checkpointDigest: string; receiptPath: string }>; qualified: true; published: false;
}
export declare function validateOnboardingExecution(config: OnboardingExecutionConfig): void;
export declare function runOnboardingExecution(config: OnboardingExecutionConfig, options: OnboardingExecutionPorts & {
  sourceRoot: string; outputDir: string; sourceIdentity: Record<string, unknown>; conversionPath: string; conversionDigest: string;
}): Promise<OnboardingExecutionResult>;
