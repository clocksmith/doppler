import { selectRuleValue } from '../../../rules/rule-registry.js';

export function resolveCapabilityKernelPathRef(configuredKernelPathRef, kernelPathSource, capabilities) {
  if (typeof configuredKernelPathRef !== 'string') {
    return configuredKernelPathRef;
  }

  const hasSubgroups = capabilities?.hasSubgroups === true;
  const allowCapabilityAutoSelection = kernelPathSource === 'model' || kernelPathSource === 'manifest';

  return selectRuleValue('inference', 'kernelPath', 'autoSelect', {
    kernelPathRef: configuredKernelPathRef,
    hasSubgroups,
    allowCapabilityAutoSelection,
  });
}
