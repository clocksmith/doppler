function hasOwnProperty(value, key) {
  return value != null && Object.prototype.hasOwnProperty.call(value, key);
}

function hasPatchEntries(value, label) {
  if (value == null) {
    return false;
  }
  if (!Array.isArray(value)) {
    throw new Error(`[ExecutionV1] ${label} must be an array when provided.`);
  }
  return value.length > 0;
}

function cloneExecutionTuple(tuple) {
  return Array.isArray(tuple) ? [...tuple] : tuple;
}

function cloneExecutionEntry(entry) {
  if (Array.isArray(entry)) {
    return cloneExecutionTuple(entry);
  }
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    return entry;
  }
  return {
    ...entry,
    layers: Array.isArray(entry.layers) ? [...entry.layers] : entry.layers,
    steps: Array.isArray(entry.steps)
      ? entry.steps.map((step) => cloneExecutionTuple(step))
      : entry.steps,
  };
}

function cloneKernelDeclaration(kernel) {
  if (!kernel || typeof kernel !== 'object' || Array.isArray(kernel)) {
    return kernel;
  }
  return {
    ...kernel,
    ...(kernel.constants ? { constants: { ...kernel.constants } } : {}),
    ...(kernel.precision ? { precision: { ...kernel.precision } } : {}),
  };
}

function cloneExecutionGraph(execution) {
  return {
    ...execution,
    kernels: Object.fromEntries(
      Object.entries(execution?.kernels ?? {}).map(
        ([key, kernel]) => [key, cloneKernelDeclaration(kernel)]
      )
    ),
    mechanismKernels: Array.isArray(execution?.mechanismKernels)
      ? [...execution.mechanismKernels]
      : execution?.mechanismKernels,
    preLayer: (execution?.preLayer ?? []).map((entry) => cloneExecutionEntry(entry)),
    decode: (execution?.decode ?? []).map((entry) => cloneExecutionEntry(entry)),
    prefill: (execution?.prefill ?? []).map((entry) => cloneExecutionEntry(entry)),
    postLayer: (execution?.postLayer ?? []).map((entry) => cloneExecutionEntry(entry)),
    policies: execution?.policies ? { ...execution.policies } : execution?.policies,
  };
}

function normalizePatchSection(section, label) {
  if (section == null) {
    return null;
  }
  if (typeof section !== 'string') {
    throw new Error(`[ExecutionV1] ${label}.section must be a string when provided.`);
  }
  const normalized = section.trim();
  if (
    normalized === 'decode'
    || normalized === 'prefill'
    || normalized === 'preLayer'
    || normalized === 'postLayer'
  ) {
    return normalized;
  }
  const lower = normalized.toLowerCase();
  if (lower === 'prelayer') {
    return 'preLayer';
  }
  if (lower === 'postlayer') {
    return 'postLayer';
  }
  throw new Error(
    `[ExecutionV1] ${label}.section must be one of decode|prefill|preLayer|postLayer.`
  );
}

function assertPatchLayersUnsupported(layers, label) {
  if (layers != null) {
    throw new Error(`[ExecutionV1] ${label}.layers is not supported yet.`);
  }
}

function applyPatchKernelAdditions(execution, addKernels) {
  for (const [index, addition] of addKernels.entries()) {
    const label = `runtime.inference.executionPatch.addKernels[${index}]`;
    if (!addition || typeof addition !== 'object' || Array.isArray(addition)) {
      throw new Error(`[ExecutionV1] ${label} must be an object.`);
    }
    const key = typeof addition.key === 'string' ? addition.key.trim() : '';
    if (!key) {
      throw new Error(`[ExecutionV1] ${label}.key must be a non-empty string.`);
    }
    if (hasOwnProperty(execution.kernels, key)) {
      throw new Error(`[ExecutionV1] ${label}.key "${key}" already exists in execution.kernels.`);
    }
    if (!addition.kernel || typeof addition.kernel !== 'object' || Array.isArray(addition.kernel)) {
      throw new Error(`[ExecutionV1] ${label}.kernel must be an object.`);
    }
    execution.kernels[key] = cloneKernelDeclaration(addition.kernel);
  }
}

function patchTuple(tuple, patch, label) {
  const patched = [...tuple];
  if (hasOwnProperty(patch, 'kernelKey') && patch.kernelKey !== undefined) {
    const kernelKey = typeof patch.kernelKey === 'string' ? patch.kernelKey.trim() : '';
    if (!kernelKey) {
      throw new Error(`[ExecutionV1] ${label}.kernelKey must be a non-empty string.`);
    }
    patched[1] = kernelKey;
  }
  if (hasOwnProperty(patch, 'weights') && patch.weights !== undefined) {
    const weights = typeof patch.weights === 'string' ? patch.weights.trim() : '';
    if (!weights) {
      throw new Error(`[ExecutionV1] ${label}.weights must be a non-empty string.`);
    }
    if (patched.length > 2) {
      patched[2] = weights;
    } else {
      patched.push(weights);
    }
  }
  return patched;
}

function applyPatchSetToEntries(entries, patch, label) {
  let matchCount = 0;
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    if (Array.isArray(entry)) {
      if (entry[0] !== patch.op) {
        continue;
      }
      entries[index] = patchTuple(entry, patch, label);
      matchCount += 1;
      continue;
    }
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.steps)) {
      continue;
    }
    for (let stepIndex = 0; stepIndex < entry.steps.length; stepIndex += 1) {
      const step = entry.steps[stepIndex];
      if (!Array.isArray(step) || step[0] !== patch.op) {
        continue;
      }
      entry.steps[stepIndex] = patchTuple(step, patch, label);
      matchCount += 1;
    }
  }
  return matchCount;
}

function applyPatchSets(execution, setPatches) {
  const defaultSections = ['preLayer', 'decode', 'prefill', 'postLayer'];
  for (const [index, patch] of setPatches.entries()) {
    const label = `runtime.inference.executionPatch.set[${index}]`;
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) {
      throw new Error(`[ExecutionV1] ${label} must be an object.`);
    }
    const op = typeof patch.op === 'string' ? patch.op.trim() : '';
    if (!op) {
      throw new Error(`[ExecutionV1] ${label}.op must be a non-empty string.`);
    }
    assertPatchLayersUnsupported(patch.layers, label);
    if (!hasOwnProperty(patch, 'kernelKey') && !hasOwnProperty(patch, 'weights')) {
      throw new Error(`[ExecutionV1] ${label} must provide kernelKey or weights.`);
    }
    const normalizedPatch = {
      ...patch,
      op,
      section: normalizePatchSection(patch.section, label),
    };
    const sections = normalizedPatch.section ? [normalizedPatch.section] : defaultSections;
    let matchCount = 0;
    for (const section of sections) {
      matchCount += applyPatchSetToEntries(execution[section] ?? [], normalizedPatch, label);
    }
    if (matchCount === 0) {
      throw new Error(`[ExecutionV1] ${label} did not match any execution step.`);
    }
  }
}

function assertUnsupportedPatchLists(executionPatch) {
  if (hasPatchEntries(executionPatch.remove, 'runtime.inference.executionPatch.remove')) {
    throw new Error('[ExecutionV1] runtime.inference.executionPatch.remove is not supported yet.');
  }
  if (hasPatchEntries(executionPatch.add, 'runtime.inference.executionPatch.add')) {
    throw new Error('[ExecutionV1] runtime.inference.executionPatch.add is not supported yet.');
  }
}

export function applyExecutionPatch(execution, executionPatch) {
  if (executionPatch == null) {
    return execution;
  }
  if (typeof executionPatch !== 'object' || Array.isArray(executionPatch)) {
    throw new Error('[ExecutionV1] runtime.inference.executionPatch must be an object.');
  }
  assertUnsupportedPatchLists(executionPatch);
  const hasKernelAdditions = hasPatchEntries(
    executionPatch.addKernels,
    'runtime.inference.executionPatch.addKernels'
  );
  const hasSets = hasPatchEntries(executionPatch.set, 'runtime.inference.executionPatch.set');
  if (!hasKernelAdditions && !hasSets) {
    return execution;
  }
  const patchedExecution = cloneExecutionGraph(execution);
  if (hasKernelAdditions) {
    applyPatchKernelAdditions(patchedExecution, executionPatch.addKernels);
  }
  if (hasSets) {
    applyPatchSets(patchedExecution, executionPatch.set);
  }
  return patchedExecution;
}
