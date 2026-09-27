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
