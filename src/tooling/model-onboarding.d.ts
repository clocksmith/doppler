export interface OnboardingInput { path: string; digest: `sha256:${string}` }
export interface ModelOnboardingConfig {
  schema: 'doppler.model-onboarding/v1';
  sourceSpec: OnboardingInput;
  vocabulary: OnboardingInput;
  entryPointIds: string[];
  lineage: { recipe: OnboardingInput; template: OnboardingInput } | null;
}
export interface ModelOnboardingResult {
  schema: 'doppler.model-onboarding-result/v1' | 'doppler.model-onboarding-result/v2';
  inputDigest: string;
  status: 'blocked' | 'recipe-required' | 'candidate-materialized' | 'capsule-qualified';
  outputs: Record<string, { path: string; digest: string }>;
  sourceIdentity: import('../config/model-ir-v2.js').ModelIRV2['sourceIdentity'];
  manualRequirements: Array<Record<string, unknown>>;
  qualified: boolean;
  published: false;
}
/** Explicit execution ports may complete the pinned stages. Revalidates retained inputs and outputs on resume; never publishes. */
export declare function runModelOnboarding(config: ModelOnboardingConfig, options: {
  sourceRoot: string; outputDir: string;
  execution?: import('./model-onboarding-execution.js').OnboardingExecutionPorts & {
    config: import('./model-onboarding-execution.js').OnboardingExecutionConfig;
  };
}): Promise<ModelOnboardingResult>;
