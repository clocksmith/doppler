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

/**
 * Map of `operation/variant` to the declared replacement program: a registry variant whose
 * `variantMetadata.replacesMechanism` names that mechanism and whose shader, entry and
 * specialization constants the graph's mechanismKernels declare. Empty without a graph.
 */
export declare function resolveDeclaredMechanismReplacements(
  graph: import('./schema/execution-v1.schema.js').ExecutionV1GraphSchema | null | undefined
): Readonly<Record<string, string>>;

/** Throw when a declared kernel's registry norm-gate activation differs from the manifest's. */
export declare function assertLinearNormGateClosure(
  graph: import('./schema/execution-v1.schema.js').ExecutionV1GraphSchema,
  normGateActivation: 'silu' | 'sigmoid'
): void;

/**
 * Require the paired mechanism closure. `normGateMechanisms` lists the
 * activation-specific [operation, variant] pairs selected by
 * `kernels.gatedDeltaRecurrent` rules (see paired-execution.js).
 */
export declare function assertPairedExecutionMechanismClosure(
  graph: import('./schema/execution-v1.schema.js').ExecutionV1GraphSchema,
  normGateMechanisms: ReadonlyArray<readonly [string, string]>
): void;
