/**
 * Kernel Path Loader
 *
 * Loads and resolves kernel path configurations.
 *
 * @module config/kernel-path-loader
 */

import { DEFAULT_ENTRY } from './schema/kernel-path.schema.js';
import { KERNEL_CONFIGS } from '../gpu/kernels/utils.js';

// =============================================================================
// Built-in Kernel Paths (imported at build time)
// =============================================================================

import gemma2Q4kFused from './presets/kernel-paths/gemma2-q4k-fused.json' with { type: 'json' };
import gemma2Q4kDequantF32 from './presets/kernel-paths/gemma2-q4k-dequant-f32.json' with { type: 'json' };
import gemma2Q4kDequantF16 from './presets/kernel-paths/gemma2-q4k-dequant-f16.json' with { type: 'json' };
import gemma2F16Native from './presets/kernel-paths/gemma2-f16-native.json' with { type: 'json' };

/** @type {Record<string, import('./schema/kernel-path.schema.js').KernelPathSchema>} */
const KERNEL_PATH_REGISTRY = {
  // Gemma 2 Q4K variants
  'gemma2-q4k-fused': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kFused),
  'gemma2-q4k-dequant-f32': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kDequantF32),
  'gemma2-q4k-dequant-f16': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kDequantF16),

  // Gemma 2 F16 native
  'gemma2-f16-native': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2F16Native),

  // Aliases for generic access (model-agnostic)
  'q4k-fused': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kFused),
  'q4k-dequant-f32': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kDequantF32),
  'q4k-dequant-f16': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kDequantF16),
  'f16-native': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2F16Native),

  // Semantic aliases
  'q4k-safe': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kDequantF32), // Max compatibility, no fusion
  'q4k-fast': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kFused), // Best throughput
  'q4k-balanced': /** @type {import('./schema/kernel-path.schema.js').KernelPathSchema} */ (gemma2Q4kDequantF16), // Good speed/accuracy tradeoff
};

// =============================================================================
// Public API
// =============================================================================

/**
 * Get a kernel path by ID.
 * @param {string} id
 * @returns {import('./schema/kernel-path.schema.js').KernelPathSchema | null}
 */
export function getKernelPath(id) {
  return KERNEL_PATH_REGISTRY[id] ?? null;
}

/**
 * List all available kernel path IDs.
 * @returns {string[]}
 */
export function listKernelPaths() {
  return Object.keys(KERNEL_PATH_REGISTRY);
}

/**
 * Resolve a kernel path reference to a full schema.
 * @param {import('./schema/kernel-path.schema.js').KernelPathRef} ref
 * @returns {import('./schema/kernel-path.schema.js').KernelPathSchema}
 */
export function resolveKernelPath(ref) {
  if (typeof ref === 'string') {
    const path = getKernelPath(ref);
    if (!path) {
      throw new Error(`Unknown kernel path: ${ref}. Available: ${listKernelPaths().join(', ')}`);
    }
    return path;
  }
  return ref;
}

/**
 * Auto-select kernel path based on model quantization and capabilities.
 *
 * Selection priority:
 * - F16/BF16 models: use f16-native
 * - Q4K with subgroups: use fused path (best throughput)
 * - Q4K with F16 support: use dequant-f16 (balanced)
 * - Q4K fallback: use dequant-f32 (max compatibility)
 * @param {string | null} quantization
 * @param {string} modelFamily
 * @param {{ hasSubgroups?: boolean; hasF16?: boolean }} [capabilities]
 * @returns {import('./schema/kernel-path.schema.js').KernelPathSchema}
 */
export function autoSelectKernelPath(
  quantization,
  modelFamily,
  capabilities = {}
) {
  const family = modelFamily.toLowerCase();
  const familyPrefix =
    family.includes('gemma3') ? 'gemma3' :
      family.includes('gemma') ? 'gemma2' :
        null;

  /** @param {string} suffix @returns {import('./schema/kernel-path.schema.js').KernelPathSchema} */
  const resolveAutoPath = (suffix) => {
    if (familyPrefix) {
      const prefixed = getKernelPath(`${familyPrefix}-${suffix}`);
      if (prefixed) return prefixed;
    }
    return resolveKernelPath(suffix);
  };

  const quantLower = quantization?.toLowerCase() ?? '';
  if (!quantization || quantLower === 'f16' || quantLower === 'bf16') {
    return resolveAutoPath('f16-native');
  }

  if (quantization.toLowerCase().includes('q4')) {
    // Prefer fused if subgroups available
    if (capabilities.hasSubgroups) {
      return resolveAutoPath('q4k-fused');
    }
    // Use F16 dequant if F16 math is available
    if (capabilities.hasF16) {
      return resolveAutoPath('q4k-dequant-f16');
    }
    // Fallback to F32 (safest, most compatible)
    return resolveAutoPath('q4k-dequant-f32');
  }

  // Default fallback
  return resolveAutoPath('q4k-dequant-f32');
}

// =============================================================================
// Step Resolution
// =============================================================================

/**
 * Resolve layer index template in weight references.
 * Replaces {L} with the actual layer index.
 * @param {string} template
 * @param {number} layerIndex
 * @returns {string}
 */
export function resolveWeightRef(template, layerIndex) {
  return template.replace(/\{L\}/g, String(layerIndex));
}

/**
 * Get steps for a specific layer, applying any overrides.
 * @param {import('./schema/kernel-path.schema.js').KernelPathSchema} path
 * @param {number} layerIndex
 * @param {'prefill' | 'decode'} phase
 * @returns {import('./schema/kernel-path.schema.js').KernelStepSchema[]}
 */
export function getLayerSteps(
  path,
  layerIndex,
  phase
) {
  // Check for layer-specific overrides
  if (path.layerOverrides) {
    for (const override of path.layerOverrides) {
      if (override.layers.includes(layerIndex)) {
        return override.steps;
      }
    }
  }

  // Use phase-specific or decode as fallback
  const layerPath = phase === 'prefill' && path.prefill ? path.prefill : path.decode;
  return layerPath.steps;
}

/**
 * Validate a kernel path schema.
 * @param {import('./schema/kernel-path.schema.js').KernelPathSchema} path
 * @returns {string[]}
 */
export function validateKernelPath(path) {
  /** @type {string[]} */
  const errors = [];

  if (!path.id) errors.push('Missing path id');
  if (!path.name) errors.push('Missing path name');
  if (!path.decode?.steps?.length) errors.push('Missing decode steps');

  /**
   * @param {import('./schema/kernel-path.schema.js').KernelStepSchema[]} steps
   * @param {string} context
   */
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

/** @typedef {'prefill' | 'decode'} KernelPathPhase */
/** @typedef {'layer' | 'preLayer' | 'postLayer' | 'sampling'} KernelPathSection */
/** @typedef {'runtime' | 'config' | 'model' | 'manifest' | 'auto' | 'none'} KernelPathSource */

/** @type {Record<string, { section: KernelPathSection; ops: string[] }>} */
const MATMUL_ROLE_ALIASES = {
  q_proj: { section: 'layer', ops: ['q_proj'] },
  k_proj: { section: 'layer', ops: ['k_proj'] },
  v_proj: { section: 'layer', ops: ['v_proj'] },
  qkv_proj: { section: 'layer', ops: ['qkv_proj', 'q_proj'] },
  o_proj: { section: 'layer', ops: ['o_proj'] },
  ffn_gate: { section: 'layer', ops: ['ffn_gate', 'gate_proj'] },
  ffn_up: { section: 'layer', ops: ['ffn_up', 'up_proj'] },
  ffn_down: { section: 'layer', ops: ['ffn_down', 'down_proj'] },
  ffn_gate_up: { section: 'layer', ops: ['ffn_gate_up'] },
  lm_head: { section: 'postLayer', ops: ['lm_head'] },
};

/**
 * @param {string} kernel
 * @returns {string}
 */
function normalizeKernelFile(kernel) {
  const trimmed = kernel.trim();
  if (!trimmed) return trimmed;
  const parts = trimmed.split('/');
  return parts[parts.length - 1] ?? trimmed;
}

/**
 * @param {import('./schema/kernel-path.schema.js').KernelPathSchema} path
 * @param {KernelPathSection} section
 * @param {KernelPathPhase} phase
 * @param {number} layerIndex
 * @returns {import('./schema/kernel-path.schema.js').KernelStepSchema[]}
 */
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

/**
 * @param {import('./schema/kernel-path.schema.js').KernelStepSchema[]} steps
 * @param {string} op
 * @returns {import('./schema/kernel-path.schema.js').KernelStepSchema | null}
 */
function findStepByOp(steps, op) {
  return steps.find((step) => step.op === op) ?? null;
}

/**
 * @param {keyof typeof KERNEL_CONFIGS} operation
 * @param {string} kernel
 * @param {string | undefined} entry
 * @returns {string | null}
 */
function findKernelVariant(
  operation,
  kernel,
  entry
) {
  const variants = KERNEL_CONFIGS[operation];
  if (!variants) return null;
  const normalizedKernel = normalizeKernelFile(kernel);
  const normalizedEntry = entry ?? DEFAULT_ENTRY;

  /** @type {string | null} */
  let fallbackVariant = null;
  let fallbackCount = 0;

  for (const [variant, config] of Object.entries(variants)) {
    if (config.shaderFile !== normalizedKernel) continue;
    fallbackVariant = variant;
    fallbackCount += 1;
    if (config.entryPoint === normalizedEntry) {
      return variant;
    }
  }

  if (fallbackCount === 1) {
    return fallbackVariant;
  }
  return null;
}

/**
 * @param {string | undefined} role
 * @param {KernelPathPhase} phase
 * @param {number} [layerIndex]
 * @returns {string | null}
 */
export function getKernelPathMatmulVariant(
  role,
  phase,
  layerIndex
) {
  if (!activeKernelPath || !role) return null;
  const alias = MATMUL_ROLE_ALIASES[role] ?? { section: 'layer', ops: [role] };
  const steps = getKernelPathStepsForSection(activeKernelPath, alias.section, phase, layerIndex ?? 0);
  for (const op of alias.ops) {
    const step = findStepByOp(steps, op);
    if (!step) continue;
    const variant = findKernelVariant('matmul', step.kernel, step.entry);
    if (variant) {
      return variant;
    }
  }
  return null;
}

/**
 * @param {KernelPathPhase} phase
 * @param {number} [layerIndex]
 * @returns {string | null}
 */
export function getKernelPathAttentionVariant(
  phase,
  layerIndex
) {
  if (!activeKernelPath) return null;
  const steps = getKernelPathStepsForSection(activeKernelPath, 'layer', phase, layerIndex ?? 0);
  const step = findStepByOp(steps, 'attention');
  if (!step) return null;
  return findKernelVariant('attention', step.kernel, step.entry);
}

// =============================================================================
// Active Kernel Path Registry
// =============================================================================

/** @type {import('./schema/kernel-path.schema.js').KernelPathSchema | null} */
let activeKernelPath = null;
/** @type {KernelPathSource} */
let activeKernelPathSource = 'none';

/**
 * Set the active kernel path for the current pipeline.
 * Called by Pipeline when resolving kernel path.
 * @param {import('./schema/kernel-path.schema.js').KernelPathSchema | null} path
 * @param {KernelPathSource} [source]
 */
export function setActiveKernelPath(path, source = 'none') {
  activeKernelPath = path;
  activeKernelPathSource = path ? source : 'none';
}

/**
 * Get the active kernel path.
 * @returns {import('./schema/kernel-path.schema.js').KernelPathSchema | null}
 */
export function getActiveKernelPath() {
  return activeKernelPath;
}

/**
 * @returns {KernelPathSource}
 */
export function getActiveKernelPathSource() {
  return activeKernelPathSource;
}

/**
 * @returns {boolean}
 */
export function getKernelPathStrict() {
  return activeKernelPathSource !== 'auto' && activeKernelPathSource !== 'none';
}

/**
 * Check if the active kernel path uses fused Q4K matmul.
 * Returns false if no kernel path is set (auto-selection will apply).
 * @returns {boolean}
 */
export function isActiveKernelPathFusedQ4K() {
  if (!activeKernelPath) return true; // Default to auto-selection (which prefers fused)
  /** @type {import('./schema/kernel-path.schema.js').KernelStepSchema[]} */
  const kernelSteps = [
    ...(activeKernelPath.decode?.steps ?? []),
    ...(activeKernelPath.prefill?.steps ?? []),
    ...(activeKernelPath.preLayer ?? []),
    ...(activeKernelPath.postLayer ?? []),
    ...(activeKernelPath.layerOverrides?.flatMap((override) => override.steps) ?? []),
  ];
  return kernelSteps.some((step) => step.kernel.includes('fused_matmul_q4'));
}

/**
 * Check if the active kernel path uses dequant (non-fused) Q4K matmul.
 * @returns {boolean}
 */
export function isActiveKernelPathDequant() {
  if (!activeKernelPath) return false;
  /** @type {import('./schema/kernel-path.schema.js').KernelStepSchema[]} */
  const kernelSteps = [
    ...(activeKernelPath.decode?.steps ?? []),
    ...(activeKernelPath.prefill?.steps ?? []),
    ...(activeKernelPath.preLayer ?? []),
    ...(activeKernelPath.postLayer ?? []),
    ...(activeKernelPath.layerOverrides?.flatMap((override) => override.steps) ?? []),
  ];
  return kernelSteps.some((step) => step.kernel.startsWith('matmul_'));
}

// =============================================================================
// Debug/Logging
// =============================================================================

/**
 * Format kernel path for logging.
 * @param {import('./schema/kernel-path.schema.js').KernelPathSchema} path
 * @returns {string}
 */
export function formatKernelPath(path) {
  const decodeOps = path.decode.steps.map(s => s.op).join(' → ');
  return `${path.id}: ${decodeOps}`;
}

/**
 * Get summary statistics for a kernel path.
 * @param {import('./schema/kernel-path.schema.js').KernelPathSchema} path
 * @returns {{ decodeSteps: number; prefillSteps: number; uniqueKernels: number; hasLayerOverrides: boolean }}
 */
export function getKernelPathStats(path) {
  /** @type {Set<string>} */
  const allKernels = new Set();

  /** @param {import('./schema/kernel-path.schema.js').KernelStepSchema[]} steps */
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
