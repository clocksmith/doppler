import { chooseNullish } from './value-selection.js';

export function assertExecutionPatchShape(patch, label = 'runtime.inference.executionPatch') {
  if (patch == null) return;
  if (typeof patch !== 'object' || Array.isArray(patch)) {
    throw new Error(`[ExecutionV1] ${label} must be an object.`);
  }
  for (const key of Object.keys(patch)) {
    if (!['addKernels', 'set', 'remove', 'add'].includes(key)) {
      throw new Error(`[ExecutionV1] ${label}.${key} is unknown; expected addKernels, set, remove, or add.`);
    }
  }
}

export function mergeExecutionPatchLists(basePatch, overridePatch) {
  assertExecutionPatchShape(basePatch);
  assertExecutionPatchShape(overridePatch);
  const base = basePatch ?? {};
  const override = overridePatch ?? {};
  return {
    addKernels: chooseNullish(override.addKernels, chooseNullish(base.addKernels, [])),
    set: chooseNullish(override.set, chooseNullish(base.set, [])),
    remove: chooseNullish(override.remove, chooseNullish(base.remove, [])),
    add: chooseNullish(override.add, chooseNullish(base.add, [])),
  };
}
