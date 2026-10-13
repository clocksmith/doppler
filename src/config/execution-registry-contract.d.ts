import type { RuleRegistry } from '../rules/rule-registry.js';
import type { KernelRegistry } from './kernel-registry-contract.js';
import type { TargetPlan } from './target-plan.js';
export interface ExecutionRegistries {
  readonly ruleRegistry: RuleRegistry;
  readonly kernelRegistry: KernelRegistry;
}
export interface ResolvedExecutionRegistries extends ExecutionRegistries {
  readonly identity: Readonly<{ rules: string; kernels: string }>;
}
export declare function resolveExecutionRegistries(registries?: ExecutionRegistries | null): ResolvedExecutionRegistries | null;
export declare function assertExecutionRegistriesAccepted(targetPlan: TargetPlan, registries: ResolvedExecutionRegistries | null): void;
export declare function assertExecutionMechanismClosure(
  graph: import('./schema/execution-v1.schema.js').ExecutionV1GraphSchema,
  steps: import('./schema/execution-v1.schema.js').ExecutionV1ExpandedStepSchema[]
): void;

export declare function assertPairedExecutionMechanismClosure(
  graph: import('./schema/execution-v1.schema.js').ExecutionV1GraphSchema
): void;
