import policy from './pack-adapters.json' with { type: 'json' };
import kernels from './kernels/registry.json' with { type: 'json' };
const assert = (ok, message) => { if (!ok) throw new Error(`Pack adapter: ${message}`); };

export function validatePackAdapterExecution(plan) {
  const declaration = plan.adapterExecution;
  assert(declaration?.schema === 'doppler.pack-adapter-execution/v1', 'selected TargetPlan must explicitly declare adapter execution');
  assert(Number.isSafeInteger(declaration.maxAdapters) && declaration.maxAdapters > 0
    && declaration.maxAdapters <= policy.maximumAdapters && declaration.combination === policy.combination, 'unsupported composition');
  assert(Array.isArray(declaration.formats) && declaration.formats.length > 0
    && declaration.formats.every(format => policy.formats.includes(format)), 'unsupported format');
  assert(Array.isArray(declaration.operations) && declaration.operations.length > 0
    && declaration.operations.every(operation => plan.qualification.some(row =>
      (row.operation === undefined ? policy.legacyQualificationOperation : row.operation) === operation)), 'invalid adapter operations');
  assert(Array.isArray(declaration.kernelModules) && declaration.kernelModules.length > 0
    && new Set(declaration.kernelModules).size === declaration.kernelModules.length, 'declared adapter kernel closure required');
  const closure = plan.initialExecutionIdentity?.kernelClosure;
  assert(Array.isArray(closure), 'adapter execution requires an exact initial kernel closure');
  const selected = declaration.kernelModules.map(id => closure.find(row => row.moduleId === id));
  assert(selected.every(Boolean), 'adapter kernel outside signed execution closure');
  for (const operation of policy.requiredKernelOperations) {
    const variants = Object.values(kernels.operations[operation].variants);
    assert(selected.some(row => variants.some(variant => variant.wgsl === row.file && variant.entryPoint === row.entry)),
      `missing declared ${operation} kernel for adapter execution`);
  }
  return declaration;
}
