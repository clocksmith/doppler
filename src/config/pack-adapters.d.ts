import type { TargetPlan } from './target-plan.js';
export interface PackAdapterExecutionDeclaration {
  schema: 'doppler.pack-adapter-execution/v1';
  maxAdapters: number;
  combination: 'single';
  formats: string[];
  operations: string[];
  kernelModules: string[];
}
export interface PackExecutionAdapter {
  readonly schema: 'doppler.pack-adapter/v1';
  readonly identity: string;
  readonly baseModel: { readonly modelId: string; readonly semanticRoot: string; readonly envelopeDigest: string; readonly artifactClosureDigest: string };
  readonly format: 'peft_safetensors';
  readonly manifest: Readonly<Record<string, unknown>>;
  readonly artifact: { readonly artifactId: string; readonly role: 'lora-weights'; readonly path: string; readonly hash: string; readonly sizeBytes: number };
}
export const PACK_ADAPTER_POLICY: Readonly<{ schema: string; legacyAdapterSet: readonly []; legacyQualificationOperation: 'generate'; formats: readonly string[];
  maximumAdapters: number; combination: string; requiredKernelOperations: readonly string[] }>;
export function validatePackAdapterExecution(plan: TargetPlan): PackAdapterExecutionDeclaration;
export function resolvePackAdapterSet(input: unknown, context: { pack: PackExecutionAdapter['baseModel']; targetPlan: TargetPlan; operation: string }): readonly PackExecutionAdapter[];
