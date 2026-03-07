import {
  getKernelPathActivationDtype,
  getKernelPathOutputDtype,
  getKernelPathKVDtype,
  resolveKernelPath,
} from '../config/kernel-path-loader.js';
import {
  DEFAULT_EXECUTION_V0_POLICIES,
  EXECUTION_V0_SCHEMA_ID,
} from '../config/schema/execution-v0.schema.js';
import { buildKernelRefFromKernelEntry } from '../config/kernels/kernel-ref.js';

function normalizeKernelDtype(value) {
  if (!value) return null;
  const lower = String(value).trim().toLowerCase();
  if (!lower) return null;
  if (lower === 'bf16' || lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'fp32' || lower === 'float32') return 'f32';
  return lower;
}

function cloneJson(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function sanitizeStepIdToken(value) {
  return String(value ?? 'step')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'step';
}

function buildKernelRef(step) {
  const kernel = String(step?.kernel ?? '').trim();
  const entry = String(step?.entry ?? 'main').trim() || 'main';
  if (!kernel) {
    return null;
  }
  return buildKernelRefFromKernelEntry(kernel, entry);
}

function toExecutionStep(step, phase, section, index) {
  const opToken = sanitizeStepIdToken(step?.op);
  const kernelRef = buildKernelRef(step);
  return {
    id: `${section}_${phase}_${index}_${opToken}`,
    phase,
    section,
    op: step.op,
    src: 'state',
    dst: 'state',
    ...(kernelRef ? { kernelRef } : {}),
    ...(step.kernel ? { kernel: step.kernel } : {}),
    ...(step.entry ? { entry: step.entry } : {}),
    ...(step.weights ? { weights: step.weights } : {}),
    ...(step.constants ? { constants: cloneJson(step.constants) } : {}),
    layers: 'all',
  };
}

function appendSectionSteps(target, steps, phase, section, nextIndexRef) {
  for (const step of steps ?? []) {
    if (!step || typeof step !== 'object' || typeof step.op !== 'string') {
      continue;
    }
    target.push(toExecutionStep(step, phase, section, nextIndexRef.value));
    nextIndexRef.value += 1;
  }
}

function buildExecutionSteps(kernelPath) {
  const steps = [];
  const nextIndexRef = { value: 0 };

  appendSectionSteps(steps, kernelPath.preLayer ?? [], 'both', 'preLayer', nextIndexRef);
  appendSectionSteps(steps, kernelPath.decode?.steps ?? [], 'decode', 'layer', nextIndexRef);

  const prefillSteps = (kernelPath.prefill?.steps?.length ?? 0) > 0
    ? kernelPath.prefill.steps
    : (kernelPath.decode?.steps ?? []);
  appendSectionSteps(steps, prefillSteps, 'prefill', 'layer', nextIndexRef);

  appendSectionSteps(steps, kernelPath.postLayer ?? [], 'both', 'postLayer', nextIndexRef);
  appendSectionSteps(steps, kernelPath.sampling ?? [], 'decode', 'sampling', nextIndexRef);

  return steps;
}

function buildKernelProfiles(steps) {
  const profilesByKey = new Map();
  for (const step of steps) {
    if (!step.kernelRef) continue;
    const key = `${step.kernelRef.id}|${step.kernelRef.version}|${step.kernelRef.digest}`;
    if (profilesByKey.has(key)) continue;
    profilesByKey.set(key, {
      kernelRef: cloneJson(step.kernelRef),
    });
  }
  return [...profilesByKey.values()];
}

function buildSessionDefaults(kernelPath) {
  const activationDtype = normalizeKernelDtype(getKernelPathActivationDtype(kernelPath));
  if (!activationDtype) {
    throw new Error('execution-v0 manifest: kernel path is missing activationDtype.');
  }
  const outputDtype = normalizeKernelDtype(getKernelPathOutputDtype(kernelPath)) ?? activationDtype;
  const kvDtype = normalizeKernelDtype(getKernelPathKVDtype(kernelPath)) ?? activationDtype;
  return {
    compute: {
      defaults: {
        activationDtype,
        mathDtype: activationDtype,
        accumDtype: 'f32',
        outputDtype,
      },
      kernelProfiles: [],
    },
    kvcache: {
      kvDtype,
    },
    decodeLoop: null,
  };
}

export function buildExecutionV0FromKernelPath(kernelPathRef) {
  if (!kernelPathRef) {
    return null;
  }
  const kernelPath = resolveKernelPath(kernelPathRef);
  const steps = buildExecutionSteps(kernelPath);
  if (steps.length === 0) {
    return null;
  }
  const sessionDefaults = buildSessionDefaults(kernelPath);
  sessionDefaults.compute.kernelProfiles = buildKernelProfiles(steps);
  return {
    schema: EXECUTION_V0_SCHEMA_ID,
    sessionDefaults,
    execution: {
      steps,
      policies: { ...DEFAULT_EXECUTION_V0_POLICIES },
    },
  };
}
