import contract from './capsule-token-selection.json' with { type: 'json' };
import { getKernelConfig } from './kernel-registry-contract.js';

export const CAPSULE_TOKEN_SELECTION_CONTRACT = Object.freeze({
  ...contract, requiredKernels: Object.freeze(contract.requiredKernels.map(row => Object.freeze({ ...row }))),
});

export function validateCapsuleTokenSelection(plan, modules) {
  const declaration = plan.tokenSelection;
  const require = (condition, message) => {
    if (!condition) throw new Error(`Capsule token selection: ${message}`);
  };
  require(plan.schema === 'doppler.target-plan/v2', 'GPU token selection requires TargetPlan v2.');
  require(declaration && Object.keys(declaration).every(key =>
    ['schema', 'generationContract', 'logitsDtype', 'kernelModules'].includes(key)), 'invalid declaration fields.');
  require(declaration.schema === contract.declarationSchema, 'unsupported declaration schema.');
  require(declaration.generationContract === contract.generationContract, 'unsupported generation contract.');
  require(declaration.logitsDtype === contract.logitsDtype, 'this recipe requires f32 logits.');
  require(plan.initialExecutionIdentity?.dtypeLane?.output === contract.logitsDtype, 'the initial execution identity must declare f32 output.');
  require(['prefill', 'decode'].every(phase => plan.phases?.[phase]?.length === 1
    && plan.phases[phase][0].kind === 'program-phase'), 'the recipe requires one program command per generation phase.');
  require(plan.qualification?.some(row => row.operation === undefined || row.operation === 'generate'), 'generation qualification is required.');
  const ids = declaration.kernelModules;
  require(Array.isArray(ids) && ids.length === contract.requiredKernels.length
    && ids.every(id => typeof id === 'string' && id.length > 0)
    && new Set(ids).size === ids.length, 'an exact, unique kernel module closure is required.');
  require(ids.every(id => plan.kernelClosure?.some(row => row.moduleId === id)),
    'sampling kernel is outside the selected TargetPlan closure.');
  if (modules !== undefined) {
    const selected = ids.map(id => modules.find(row => row.id === id));
    require(selected.every(Boolean), 'sampling kernel is outside the Capsule closure.');
    require(contract.requiredKernels.every(({ operation, variant }) =>
      selected.some(row => row.file === getKernelConfig(operation, variant).shaderFile)),
      'the recipe must bind every required sampling shader.');
    require(selected.every(row => plan.kernelClosure.some(ref => ref.moduleId === row.id
      && ref.digest === row.digest && ref.sourceHash === row.sourceHash)), 'sampling shader identity mismatch.');
  }
  return declaration;
}
