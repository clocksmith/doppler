import type { TargetPlan } from './target-plan.js';
export interface CapsuleAdapterExecutionDeclaration {
  schema: 'doppler.capsule-adapter-execution/v1';
  maxAdapters: number;
  combination: 'single';
  formats: string[];
  operations: string[];
  kernelModules: string[];
}
export interface CapsuleExecutionAdapter {
  readonly schema: 'doppler.capsule-adapter/v1';
  readonly identity: string;
  readonly baseModel: { readonly modelId: string; readonly semanticRoot: string; readonly envelopeDigest: string; readonly artifactClosureDigest: string };
  readonly format: 'peft_safetensors';
  readonly manifest: Readonly<Record<string, unknown>>;
  readonly artifact: { readonly artifactId: string; readonly role: 'lora-weights'; readonly path: string; readonly hash: string; readonly sizeBytes: number };
}
export const CAPSULE_ADAPTER_POLICY: Readonly<{ schema: string; legacyAdapterSet: readonly []; legacyQualificationOperation: 'generate'; formats: readonly string[];
  maximumAdapters: number; combination: string; requiredKernelOperations: readonly string[] }>;
export function validateCapsuleAdapterExecution(plan: TargetPlan): CapsuleAdapterExecutionDeclaration;
export function resolveCapsuleAdapterSet(input: unknown, context: { capsule: CapsuleExecutionAdapter['baseModel']; targetPlan: TargetPlan; operation: string }): readonly CapsuleExecutionAdapter[];
