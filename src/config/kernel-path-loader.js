import { DEFAULT_ENTRY } from './schema/kernel-path.schema.js';
import { KERNEL_CONFIGS } from '../gpu/kernels/utils.js';
import { mergeKernelPathPolicy } from './merge-helpers.js';

// =============================================================================
// Public API (registry removed — Phase 3)
// =============================================================================

export function getKernelPathActivationDtype(path) {
  if (!path?.activationDtype) return null;
  return path.activationDtype;
}

export function getKernelPathOutputDtype(path) {
  if (!path?.outputDtype) return null;
  return path.outputDtype;
}

export function getKernelPathKVDtype(path) {
  if (!path) return null;
  if (path.kvDtype) return path.kvDtype;
  if (path.activationDtype) return path.activationDtype;
  return null;
}

/**
 * Resolve a kernel path reference to a full schema object.
 * After the registry removal (Phase 3), only object refs are supported.
 * String-based registry lookups are no longer available.
 */
export function resolveKernelPath(ref) {
  if (typeof ref === 'string') {
    throw new Error(
      `String kernel path ids are no longer supported (registry removed). Got: "${ref}". ` +
      'Use an inline kernel path object or execution graph transforms instead.'
    );
  }
  return ref;
}

// =============================================================================
// Step Resolution
// =============================================================================

export function resolveWeightRef(template, layerIndex) {
  return template.replace(/\{L\}/g, String(layerIndex));
}

export function getLayerSteps(
  path,
  layerIndex,
  phase
) {
  const resolveOverrideSteps = (override) => {
    const phaseSteps = phase === 'prefill'
      ? override.prefill?.steps
      : override.decode?.steps;
    if (Array.isArray(phaseSteps) && phaseSteps.length > 0) {
      return phaseSteps;
    }
    if (Array.isArray(override.steps) && override.steps.length > 0) {
      return override.steps;
    }
    return null;
  };

  if (path.layerOverrides) {
    for (const override of path.layerOverrides) {
      if (override.layers.includes(layerIndex)) {
        const overrideSteps = resolveOverrideSteps(override);
        if (overrideSteps) {
          return overrideSteps;
        }
        break;
      }
    }
  }

  // Use phase-specific or decode as fallback
  const layerPath = phase === 'prefill' && path.prefill ? path.prefill : path.decode;
  return layerPath.steps;
}

export function validateKernelPath(path) {
  const errors = [];

  if (!path.id) errors.push('Missing path id');
  if (!path.name) errors.push('Missing path name');
  if (!path.activationDtype) errors.push('Missing activationDtype');
  if (!path.decode?.steps?.length) errors.push('Missing decode steps');

  const validateSteps = (steps, context) => {
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];
      if (!step.op) errors.push(`${context}[${i}]: missing op`);
      if (!step.kernel) errors.push(`${context}[${i}]: missing kernel`);
    }
  };

  if (path.decode?.steps) validateSteps(path.decode.steps, 'decode');
  if (path.prefill?.steps) validateSteps(path.prefill.steps, 'prefill');
  if (path.preLayer) validateSteps(path.preLayer, 'preLayer');
  if (path.postLayer) validateSteps(path.postLayer, 'postLayer');
  if (path.sampling) validateSteps(path.sampling, 'sampling');

  return errors;
}

// =============================================================================
// Kernel Path Variant Resolution
// =============================================================================

const MATMUL_STEP_ROLE_ALIASES = {
  q_proj: { section: 'layer', ops: ['q_proj'] },
  k_proj: { section: 'layer', ops: ['k_proj'] },
  v_proj: { section: 'layer', ops: ['v_proj'] },
  qkv_proj: { section: 'layer', ops: ['qkv_proj', 'q_proj'] },
  linear_qkv_proj: { section: 'layer', ops: ['qkv_proj', 'q_proj'] },
  linear_z_proj: { section: 'layer', ops: ['q_proj'] },
  o_proj: { section: 'layer', ops: ['o_proj'] },
  linear_out_proj: { section: 'layer', ops: ['o_proj'] },
  ffn_gate: { section: 'layer', ops: ['ffn_gate', 'gate_proj'] },
  ffn_up: { section: 'layer', ops: ['ffn_up', 'up_proj'] },
  ffn_down: { section: 'layer', ops: ['ffn_down', 'down_proj'] },
  ffn_gate_up: { section: 'layer', ops: ['ffn_gate_up'] },
  lm_head: { section: 'postLayer', ops: ['lm_head'] },
};

const MATMUL_PRECISION_ROLE_ALIASES = {
  ...MATMUL_STEP_ROLE_ALIASES,
  linear_a_proj: { section: 'layer', ops: ['q_proj'] },
  linear_b_proj: { section: 'layer', ops: ['q_proj'] },
};
const FUSED_FFN_PRECISION_FALLBACK_ROLES = new Set([
  'ffn_gate',
  'ffn_up',
  'ffn_down',
  'ffn_gate_up',
]);

function normalizeKernelFile(kernel) {
  const trimmed = kernel.trim();
  if (!trimmed) return trimmed;
  const parts = trimmed.split('/');
  return parts[parts.length - 1] ?? trimmed;
}

function getKernelPathStepsForSection(
  path,
  section,
  phase,
  layerIndex
) {
  switch (section) {
    case 'preLayer':
      return path.preLayer ?? [];
    case 'postLayer':
      return path.postLayer ?? [];
    case 'sampling':
      return path.sampling ?? [];
    case 'layer':
    default:
      return getLayerSteps(path, layerIndex, phase);
  }
}

function findStepByOp(steps, op) {
  return steps.find((step) => step.op === op) ?? null;
}

function pickOverrideConstants(constants, overrideKeys) {
  if (!constants || overrideKeys.size === 0) return {};
  const selected = {};
  for (const key of overrideKeys) {
    if (constants[key] !== undefined) {
      selected[key] = constants[key];
    }
  }
  return selected;
}

function overridesEqual(a, b) {
  const aKeys = Object.keys(a);
  const bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) return false;
  for (const key of aKeys) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

function findKernelVariant(
  operation,
  kernel,
  entry,
  phase,
  constants
) {
  const variants = KERNEL_CONFIGS[operation];
  if (!variants) return null;
  const normalizedKernel = normalizeKernelFile(kernel);
  const normalizedEntry = entry ?? DEFAULT_ENTRY;

  const entryMatches = [];
  let fallbackVariant = null;
  let fallbackCount = 0;

  for (const [variant, config] of Object.entries(variants)) {
    if (config.shaderFile !== normalizedKernel) continue;
    fallbackVariant = variant;
    fallbackCount += 1;
    if (config.entryPoint === normalizedEntry) {
      entryMatches.push({ variant, config });
    }
  }

  if (entryMatches.length === 1) {
    return entryMatches[0].variant;
  }
  if (entryMatches.length > 1) {
    const overrideKeys = new Set();
    for (const { config } of entryMatches) {
      const keys = Object.keys(config.wgslOverrides ?? {});
      for (const key of keys) overrideKeys.add(key);
    }
    if (overrideKeys.size > 0) {
      const requestedOverrides = pickOverrideConstants(constants, overrideKeys);
      const overrideMatches = entryMatches.filter(({ config }) =>
        overridesEqual(config.wgslOverrides ?? {}, requestedOverrides)
      );
      if (overrideMatches.length === 1) {
        return overrideMatches[0].variant;
      }
    }
  }
  if (entryMatches.length > 1 && phase) {
    const phasePrefix = `${phase}_`;
    const phaseMatch = entryMatches.find(({ variant }) => variant.startsWith(phasePrefix));
    if (phaseMatch) {
      return phaseMatch.variant;
    }
  }

  if (fallbackCount === 1) {
    return fallbackVariant;
  }
  return null;
}

export function getKernelPathMatmulVariant(
  role,
  phase,
  layerIndex,
  path = undefined
) {
  const step = getKernelPathMatmulStep(role, phase, layerIndex, path, MATMUL_STEP_ROLE_ALIASES);
  if (!step) return null;
  return findKernelVariant('matmul', step.kernel, step.entry, phase, step.constants);
}

export function getKernelPathMatmulConstants(
  role,
  phase,
  layerIndex,
  path = undefined
) {
  const step = getKernelPathMatmulStep(role, phase, layerIndex, path, MATMUL_STEP_ROLE_ALIASES);
  return step?.constants ?? null;
}

export function getKernelPathMatmulPrecision(
  role,
  phase,
  layerIndex,
  path = undefined
) {
  const step = getKernelPathMatmulStep(role, phase, layerIndex, path, MATMUL_PRECISION_ROLE_ALIASES);
  if (step?.precision) {
    return step.precision;
  }
  if (!FUSED_FFN_PRECISION_FALLBACK_ROLES.has(role)) {
    return null;
  }
  const fusedStep = getKernelPathMatmulStep('ffn', phase, layerIndex, path);
  return fusedStep?.precision ?? null;
}

export function getKernelPathStepPrecision(
  op,
  section,
  phase,
  layerIndex,
  path = undefined
) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath || !op || !section) return null;
  const steps = getKernelPathStepsForSection(lookupPath, section, phase, layerIndex ?? 0);
  const step = findStepByOp(steps, op);
  return step?.precision ?? null;
}

function getKernelPathMatmulStep(
  role,
  phase,
  layerIndex,
  path = undefined,
  aliasMap = MATMUL_STEP_ROLE_ALIASES
) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath || !role) return null;
  const alias = aliasMap[role] ?? { section: 'layer', ops: [role] };
  const steps = getKernelPathStepsForSection(lookupPath, alias.section, phase, layerIndex ?? 0);
  if (role === 'lm_head' && phase === 'prefill') {
    const prefillStep = findStepByOp(steps, 'lm_head_prefill');
    if (prefillStep) {
      return prefillStep;
    }
  }
  for (const op of alias.ops) {
    const step = findStepByOp(steps, op);
    if (step) {
      return step;
    }
  }
  return null;
}

export function getKernelPathAttentionVariant(
  phase,
  layerIndex,
  path = undefined
) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return null;
  const steps = getKernelPathStepsForSection(lookupPath, 'layer', phase, layerIndex ?? 0);
  const step = findStepByOp(steps, 'attention');
  if (!step) return null;
  return findKernelVariant('attention', step.kernel, step.entry, phase, step.constants);
}

export function getKernelPathAttentionPrecision(
  phase,
  layerIndex,
  path = undefined
) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return null;
  const steps = getKernelPathStepsForSection(lookupPath, 'layer', phase, layerIndex ?? 0);
  const step = findStepByOp(steps, 'attention');
  return step?.precision ?? null;
}

// =============================================================================
// Active Kernel Path Registry
// =============================================================================

let activeKernelPath = null;
let activeKernelPathSource = 'none';
const DEFAULT_ACTIVE_KERNEL_PATH_POLICY = {
  mode: 'locked',
  sourceScope: ['model', 'manifest'],
  onIncompatible: 'error',
};
let activeKernelPathPolicy = DEFAULT_ACTIVE_KERNEL_PATH_POLICY;

export function setActiveKernelPath(path, source = 'none', policy = undefined) {
  activeKernelPath = path;
  activeKernelPathSource = path ? source : 'none';
  activeKernelPathPolicy = mergeKernelPathPolicy(DEFAULT_ACTIVE_KERNEL_PATH_POLICY, policy);
}

export function getActiveKernelPath() {
  return activeKernelPath;
}

export function getActiveKernelPathSource() {
  return activeKernelPathSource;
}

export function getActiveKernelPathPolicy() {
  return {
    mode: activeKernelPathPolicy.mode,
    sourceScope: [...activeKernelPathPolicy.sourceScope],
    allowSources: [...activeKernelPathPolicy.sourceScope],
    onIncompatible: activeKernelPathPolicy.onIncompatible,
  };
}

export function getKernelPathStrict() {
  // Kernel-path overrides stay strict; capability-aware policy is handled at path-selection time.
  return true;
}

export function isKernelPathFusedQ4K(path = undefined) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return false;
  const kernelSteps = [
    ...(lookupPath.decode?.steps ?? []),
    ...(lookupPath.prefill?.steps ?? []),
    ...(lookupPath.preLayer ?? []),
    ...(lookupPath.postLayer ?? []),
    ...(lookupPath.layerOverrides?.flatMap((override) => [
      ...(override?.steps ?? []),
      ...(override?.decode?.steps ?? []),
      ...(override?.prefill?.steps ?? []),
    ]) ?? []),
  ];
  return kernelSteps.some((step) => step.kernel.includes('fused_matmul_q4'));
}

export function kernelPathRequiresF32MatmulWeights(path = undefined) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return false;
  const kernelSteps = [
    ...(lookupPath.decode?.steps ?? []),
    ...(lookupPath.prefill?.steps ?? []),
    ...(lookupPath.preLayer ?? []),
    ...(lookupPath.postLayer ?? []),
    ...(lookupPath.layerOverrides?.flatMap((override) => [
      ...(override?.steps ?? []),
      ...(override?.decode?.steps ?? []),
      ...(override?.prefill?.steps ?? []),
    ]) ?? []),
  ];
  return kernelSteps.some((step) => normalizeKernelFile(step.kernel) === 'matmul_f32.wgsl');
}

export function isActiveKernelPathFusedQ4K() {
  return isKernelPathFusedQ4K(activeKernelPath);
}

export function isKernelPathDequant(path = undefined) {
  const lookupPath = path === undefined ? activeKernelPath : path;
  if (!lookupPath) return false;
  const kernelSteps = [
    ...(lookupPath.decode?.steps ?? []),
    ...(lookupPath.prefill?.steps ?? []),
    ...(lookupPath.preLayer ?? []),
    ...(lookupPath.postLayer ?? []),
    ...(lookupPath.layerOverrides?.flatMap((override) => [
      ...(override?.steps ?? []),
      ...(override?.decode?.steps ?? []),
      ...(override?.prefill?.steps ?? []),
    ]) ?? []),
  ];
  return kernelSteps.some((step) => step.kernel.startsWith('matmul_'));
}

export function isActiveKernelPathDequant() {
  return isKernelPathDequant(activeKernelPath);
}

// =============================================================================
// Debug/Logging
// =============================================================================

export function formatKernelPath(path) {
  const decodeOps = path.decode.steps.map(s => s.op).join(' -> ');
  return `${path.id}: ${decodeOps}`;
}

export function getKernelPathStats(path) {
  const allKernels = new Set();

  const collectKernels = (steps) => {
    for (const step of steps) {
      allKernels.add(step.kernel);
    }
  };

  collectKernels(path.decode.steps);
  if (path.prefill) collectKernels(path.prefill.steps);
  if (path.preLayer) collectKernels(path.preLayer);
  if (path.postLayer) collectKernels(path.postLayer);
  if (path.sampling) collectKernels(path.sampling);

  return {
    decodeSteps: path.decode.steps.length,
    prefillSteps: path.prefill?.steps.length ?? path.decode.steps.length,
    uniqueKernels: allKernels.size,
    hasLayerOverrides: !!path.layerOverrides?.length,
  };
}
