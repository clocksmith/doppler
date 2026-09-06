export interface OnboardingInput { path: string; digest: `sha256:${string}` }
export interface ModelOnboardingConfig {
  schema: 'doppler.model-onboarding/v1';
  sourceSpec: OnboardingInput;
  vocabulary: OnboardingInput;
  entryPointIds: string[];
  lineage: { recipe: OnboardingInput; template: OnboardingInput } | null;
}
export interface ModelOnboardingResult {
  schema: 'doppler.model-onboarding-result/v1';
  inputDigest: string;
  status: 'blocked' | 'recipe-required' | 'candidate-materialized';
  outputs: Record<string, { path: string; digest: string }>;
  sourceIdentity: import('../config/model-ir-v2.js').ModelIRV2['sourceIdentity'];
  manualRequirements: Array<Record<string, unknown>>;
  qualified: false;
  published: false;
}
/** Local Forge orchestration only. Revalidates inputs on resume; never converts, executes or publishes implicitly. */
export declare function runModelOnboarding(config: ModelOnboardingConfig, options: {
  sourceRoot: string; outputDir: string;
}): Promise<ModelOnboardingResult>;
