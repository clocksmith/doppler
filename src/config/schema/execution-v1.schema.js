// =============================================================================
// Execution v1 Schema
// =============================================================================

export const EXECUTION_V1_SCHEMA_ID = 'doppler.execution/v1';

const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

export const DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS = {
  activationDtype: 'f16',
  mathDtype: 'f16',
  accumDtype: 'f32',
  outputDtype: 'f16',
};

export const DEFAULT_EXECUTION_V1_SESSION_DEFAULTS = {
  compute: {
    defaults: { ...DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS },
  },
  kvcache: null,
  decodeLoop: null,
};

export const DEFAULT_EXECUTION_V1_POLICIES = {
  unsupportedPrecision: 'error',
  dtypeTransition: 'require_cast_step',
  unresolvedKernel: 'error',
};

export const DEFAULT_EXECUTION_V1_PATCH = {
  set: [],
  remove: [],
  add: [],
};

export function isExecutionV1Digest(value) {
  return typeof value === 'string' && DIGEST_PATTERN.test(value);
}

export function hasExecutionV1(inference) {
  return inference?.schema === EXECUTION_V1_SCHEMA_ID;
}


function validateKernelMap(kernels, options = {}) {
  const { skipDigestValidation = false } = options;
  if (!kernels || typeof kernels !== 'object' || Array.isArray(kernels)) {
    throw new Error('execution.kernels must be a non-null object.');
  }
  for (const [key, decl] of Object.entries(kernels)) {
    if (!decl || typeof decl !== 'object') {
      throw new Error(`execution.kernels["${key}"] must be an object.`);
    }
    if (typeof decl.kernel !== 'string' || !decl.kernel.trim()) {
      throw new Error(`execution.kernels["${key}"].kernel must be a non-empty string.`);
    }
    if (typeof decl.entry !== 'string' || !decl.entry.trim()) {
      throw new Error(`execution.kernels["${key}"].entry must be a non-empty string.`);
    }
    if (!skipDigestValidation && !isExecutionV1Digest(decl.digest)) {
      throw new Error(`execution.kernels["${key}"].digest must match sha256:<64 hex chars>.`);
    }
    if (decl.constants != null && typeof decl.constants !== 'object') {
      throw new Error(`execution.kernels["${key}"].constants must be an object or null.`);
    }
  }
}


function resolveKernel(kernels, kernelKey, context) {
  const decl = kernels[kernelKey];
  if (!decl) {
    throw new Error(`${context}: kernel key "${kernelKey}" not found in execution.kernels.`);
  }
  return decl;
}


function expandTuple(tuple, kernels, phase, section, layers, context) {
  if (!Array.isArray(tuple) || tuple.length < 2 || tuple.length > 3) {
    throw new Error(`${context}: step must be [op, kernelKey] or [op, kernelKey, weights].`);
  }
  const [op, kernelKey, weights] = tuple;
  if (typeof op !== 'string' || !op.trim()) {
    throw new Error(`${context}: step op must be a non-empty string.`);
  }
  if (typeof kernelKey !== 'string' || !kernelKey.trim()) {
    throw new Error(`${context}: step kernelKey must be a non-empty string.`);
  }
  if (weights !== undefined && typeof weights !== 'string') {
    throw new Error(`${context}: step weights must be a string if provided.`);
  }
  const decl = resolveKernel(kernels, kernelKey, `${context}[${op}]`);
  return {
    op,
    kernel: decl.kernel,
    entry: decl.entry,
    digest: decl.digest,
    weights: weights ?? null,
    constants: decl.constants ?? null,
    ...(decl.precision ? { precision: decl.precision } : {}),
    layers,
    phase,
    section,
  };
}


function expandStepEntries(entries, kernels, phase, context) {
  const expanded = [];
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const entryCtx = `${context}[${i}]`;

    if (Array.isArray(entry)) {
      expanded.push(expandTuple(entry, kernels, phase, 'layer', 'all', entryCtx));
    } else if (entry && typeof entry === 'object' && Array.isArray(entry.layers) && Array.isArray(entry.steps)) {
      if (!entry.layers.every((l) => Number.isInteger(l) && l >= 0)) {
        throw new Error(`${entryCtx}: layers must be an array of non-negative integers.`);
      }
      for (let j = 0; j < entry.steps.length; j++) {
        expanded.push(expandTuple(entry.steps[j], kernels, phase, 'layer', entry.layers, `${entryCtx}.steps[${j}]`));
      }
    } else {
      throw new Error(`${entryCtx}: must be a step tuple or a { layers, steps } group.`);
    }
  }
  return expanded;
}


function expandBoundarySteps(entries, kernels, section, context) {
  const expanded = [];
  for (let i = 0; i < entries.length; i++) {
    expanded.push(expandTuple(entries[i], kernels, 'both', section, 'all', `${context}[${i}]`));
  }
  return expanded;
}


/**
 * Expand a v1 execution graph into runtime-ready expanded steps.
 *
 * @param {object} graph - The execution graph from the manifest.
 * @param {{ knownOps?: Set<string> | null, strict?: boolean }} [options]
 *   knownOps: when provided, any op not in this set triggers a warning (or
 *   throw in strict mode) with the op name and step index.
 *   strict: when true, unknown ops throw instead of warn. Default false.
 * @returns {object[]} Expanded step array.
 */
export function expandExecutionV1(graph, options = {}) {
  const { knownOps = null, strict = false, skipDigestValidation = false } = options;
  if (!graph || typeof graph !== 'object') {
    throw new Error('execution graph must be a non-null object.');
  }
  if (graph.inlineKernelPath !== undefined && typeof graph.inlineKernelPath !== 'boolean') {
    throw new Error('execution.inlineKernelPath must be a boolean when provided.');
  }

  const kernels = graph.kernels;
  validateKernelMap(kernels, { skipDigestValidation });

  const MAX_STEPS_PER_SECTION = 200;

  const preLayer = expandBoundarySteps(graph.preLayer ?? [], kernels, 'preLayer', 'execution.preLayer');
  const decode = expandStepEntries(graph.decode ?? [], kernels, 'decode', 'execution.decode');
  const prefill = expandStepEntries(graph.prefill ?? [], kernels, 'prefill', 'execution.prefill');
  const postLayer = expandBoundarySteps(graph.postLayer ?? [], kernels, 'postLayer', 'execution.postLayer');

  const sections = { preLayer, decode, prefill, postLayer };
  for (const [name, steps] of Object.entries(sections)) {
    if (steps.length > MAX_STEPS_PER_SECTION) {
      // Use console.warn directly here because the schema module is a
      // low-level config layer that should not depend on the debug module.
      console.warn(
        `[ExecutionV1] Section "${name}" has ${steps.length} expanded steps ` +
        `(max recommended: ${MAX_STEPS_PER_SECTION}). This may indicate a misconfigured execution graph.`
      );
    }
  }

  const allSteps = [...preLayer, ...decode, ...prefill, ...postLayer];

  if (knownOps) {
    const unknownOps = [];
    for (let i = 0; i < allSteps.length; i++) {
      const step = allSteps[i];
      if (!knownOps.has(step.op)) {
        unknownOps.push({ op: step.op, index: i, section: step.section });
      }
    }
    if (unknownOps.length > 0) {
      const details = unknownOps
        .map((u) => `"${u.op}" at step ${u.index} (${u.section})`)
        .join(', ');
      const message =
        `[ExecutionV1] Expanded graph contains unknown ops: ${details}. ` +
        `Known ops: ${[...knownOps].join(', ')}.`;
      if (strict) {
        throw new Error(message);
      }
      // Use console.warn directly here because the schema module is a
      // low-level config layer that should not depend on the debug module.
      // Callers in runtime code can catch the strict error or filter post-hoc.
      console.warn(message);
    }
  }

  return allSteps;
}
