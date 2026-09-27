import { isKernelRegistry } from './kernel-registry-contract.js';
import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

/** @param {import('./execution-registry-contract.js').ExecutionRegistries | null | undefined} registries */
export function resolveExecutionRegistries(registries) {
  if (registries == null) return null;
  const { ruleRegistry, kernelRegistry } = registries;
  if ((!Object.isFrozen(ruleRegistry) || typeof ruleRegistry?.selectRuleValue !== 'function'
    || !/^sha256:[0-9a-f]{64}$/.test(ruleRegistry.identity)) || !isKernelRegistry(kernelRegistry)) {
    throw new Error('Execution requires constructed rule and kernel registry instances.');
  }
  return Object.freeze({ ruleRegistry, kernelRegistry,
    identity: Object.freeze({ rules: ruleRegistry.identity, kernels: kernelRegistry.identity }) });
}

/** @param {import('./target-plan.js').TargetPlan} targetPlan
 * @param {import('./execution-registry-contract.js').ResolvedExecutionRegistries | null} registries */
export function assertExecutionRegistriesAccepted(targetPlan, registries) {
  const declared = ('initialExecutionIdentity' in targetPlan ? targetPlan.initialExecutionIdentity?.runtimeEngine?.registries : null) ?? null;
  const observed = registries?.identity ?? null;
  if (computeCanonicalSha256(declared) !== computeCanonicalSha256(observed)) {
    throw new Error('Execution registry extensions must be bound to the accepted TargetPlan initial execution identity.');
  }
  for (const variants of Object.values(registries?.kernelRegistry.validators ?? {})) {
    for (const validator of Object.values(variants)) {
      if (validator.id.startsWith('legacy:')) {
        throw new Error('Capsule execution requires explicitly identified validators; compatibility registrations are not accepted.');
      }
    }
  }
}
