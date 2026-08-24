import { chooseNullish } from './value-selection.js';

export function mergeExecutionPatchLists(basePatch, overridePatch) {
  const base = basePatch ?? {};
  const override = overridePatch ?? {};
  return {
    addKernels: chooseNullish(override.addKernels, chooseNullish(base.addKernels, [])),
    set: chooseNullish(override.set, chooseNullish(base.set, [])),
    remove: chooseNullish(override.remove, chooseNullish(base.remove, [])),
    add: chooseNullish(override.add, chooseNullish(base.add, [])),
  };
}
