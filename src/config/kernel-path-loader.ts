/**
 * Kernel Path Loader
 *
 * Loads and resolves kernel path configurations.
 *
 * @module config/kernel-path-loader
 */

import type {
  KernelPathSchema,
  KernelPathRef,
  BuiltinKernelPathId,
  KernelStepSchema,
  LayerKernelPathSchema,
} from './schema/kernel-path.schema.js';
import { DEFAULT_ENTRY } from './schema/kernel-path.schema.js';
import { KERNEL_CONFIGS } from '../gpu/kernels/utils.js';

// =============================================================================
// Built-in Kernel Paths (imported at build time)
// =============================================================================

import gemma2Q4kFused from './presets/kernel-paths/gemma2-q4k-fused.json' with { type: 'json' };
import gemma2Q4kDequantF32 from './presets/kernel-paths/gemma2-q4k-dequant-f32.json' with { type: 'json' };
import gemma2Q4kDequantF16 from './presets/kernel-paths/gemma2-q4k-dequant-f16.json' with { type: 'json' };
import gemma2F16Native from './presets/kernel-paths/gemma2-f16-native.json' with { type: 'json' };

/** Registry of built-in kernel paths */
const KERNEL_PATH_REGISTRY: Record<string, KernelPathSchema> = {
  // Gemma 2 Q4K variants
  'gemma2-q4k-fused': gemma2Q4kFused as KernelPathSchema,
  'gemma2-q4k-dequant-f32': gemma2Q4kDequantF32 as KernelPathSchema,
  'gemma2-q4k-dequant-f16': gemma2Q4kDequantF16 as KernelPathSchema,

  // Gemma 2 F16 native
  'gemma2-f16-native': gemma2F16Native as KernelPathSchema,

  // Aliases for generic access (model-agnostic)
  'q4k-fused': gemma2Q4kFused as KernelPathSchema,
  'q4k-dequant-f32': gemma2Q4kDequantF32 as KernelPathSchema,
  'q4k-dequant-f16': gemma2Q4kDequantF16 as KernelPathSchema,
  'f16-native': gemma2F16Native as KernelPathSchema,

  // Semantic aliases
  'q4k-safe': gemma2Q4kDequantF32 as KernelPathSchema, // Max compatibility, no fusion
  'q4k-fast': gemma2Q4kFused as KernelPathSchema, // Best throughput
  'q4k-balanced': gemma2Q4kDequantF16 as KernelPathSchema, // Good speed/accuracy tradeoff
};

// =============================================================================
// Public API
// =============================================================================

/**
 * Get a kernel path by ID.
 */
export function getKernelPath(id: string): KernelPathSchema | null {
  return KERNEL_PATH_REGISTRY[id] ?? null;
}

/**
 * List all available kernel path IDs.
 */
export function listKernelPaths(): string[] {
  return Object.keys(KERNEL_PATH_REGISTRY);
}

/**
 * Resolve a kernel path reference to a full schema.
 */
export function resolveKernelPath(ref: KernelPathRef): KernelPathSchema {
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
 */
export function autoSelectKernelPath(
  quantization: string | null,
  modelFamily: string,
  capabilities: { hasSubgroups?: boolean; hasF16?: boolean } = {}
): KernelPathSchema {
  const family = modelFamily.toLowerCase();
  const familyPrefix =
    family.includes('gemma3') ? 'gemma3' :
      family.includes('gemma') ? 'gemma2' :
        null;

  const resolveAutoPath = (suffix: string): KernelPathSchema => {
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
 */
export function resolveWeightRef(template: string, layerIndex: number): string {
  return template.replace(/\{L\}/g, String(layerIndex));
}

/**
 * Get steps for a specific layer, applying any overrides.
 */
export function getLayerSteps(
  path: KernelPathSchema,
  layerIndex: number,
  phase: 'prefill' | 'decode'
): KernelStepSchema[] {
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
 */
export function validateKernelPath(path: KernelPathSchema): string[] {
  const errors: string[] = [];

  if (!path.id) errors.push('Missing path id');
  if (!path.name) errors.push('Missing path name');
  if (!path.decode?.steps?.length) errors.push('Missing decode steps');

  // Validate each step
  const validateSteps = (steps: KernelStepSchema[], context: string) => {
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

export type KernelPathPhase = 'prefill' | 'decode';
export type KernelPathSection = 'layer' | 'preLayer' | 'postLayer' | 'sampling';
export type KernelPathSource = 'runtime' | 'config' | 'model' | 'manifest' | 'auto' | 'none';

const MATMUL_ROLE_ALIASES: Record<string, { section: KernelPathSection; ops: string[] }> = {
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

function normalizeKernelFile(kernel: string): string {
  const trimmed = kernel.trim();
  if (!trimmed) return trimmed;
  const parts = trimmed.split('/');
  return parts[parts.length - 1] ?? trimmed;
}

function getKernelPathStepsForSection(
  path: KernelPathSchema,
  section: KernelPathSection,
  phase: KernelPathPhase,
  layerIndex: number
): KernelStepSchema[] {
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

function findStepByOp(steps: KernelStepSchema[], op: string): KernelStepSchema | null {
  return steps.find((step) => step.op === op) ?? null;
}

function findKernelVariant(
  operation: keyof typeof KERNEL_CONFIGS,
  kernel: string,
  entry: string | undefined
): string | null {
  const variants = KERNEL_CONFIGS[operation];
  if (!variants) return null;
  const normalizedKernel = normalizeKernelFile(kernel);
  const normalizedEntry = entry ?? DEFAULT_ENTRY;

  let fallbackVariant: string | null = null;
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

export function getKernelPathMatmulVariant(
  role: string | undefined,
  phase: KernelPathPhase,
  layerIndex?: number
): string | null {
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

export function getKernelPathAttentionVariant(
  phase: KernelPathPhase,
  layerIndex?: number
): string | null {
  if (!activeKernelPath) return null;
  const steps = getKernelPathStepsForSection(activeKernelPath, 'layer', phase, layerIndex ?? 0);
  const step = findStepByOp(steps, 'attention');
  if (!step) return null;
  return findKernelVariant('attention', step.kernel, step.entry);
}

// =============================================================================
// Active Kernel Path Registry
// =============================================================================

let activeKernelPath: KernelPathSchema | null = null;
let activeKernelPathSource: KernelPathSource = 'none';

/**
 * Set the active kernel path for the current pipeline.
 * Called by Pipeline when resolving kernel path.
 */
export function setActiveKernelPath(path: KernelPathSchema | null, source: KernelPathSource = 'none'): void {
  activeKernelPath = path;
  activeKernelPathSource = path ? source : 'none';
}

/**
 * Get the active kernel path.
 */
export function getActiveKernelPath(): KernelPathSchema | null {
  return activeKernelPath;
}

export function getActiveKernelPathSource(): KernelPathSource {
  return activeKernelPathSource;
}

export function getKernelPathStrict(): boolean {
  return activeKernelPathSource !== 'auto' && activeKernelPathSource !== 'none';
}

/**
 * Check if the active kernel path uses fused Q4K matmul.
 * Returns false if no kernel path is set (auto-selection will apply).
 */
export function isActiveKernelPathFusedQ4K(): boolean {
  if (!activeKernelPath) return true; // Default to auto-selection (which prefers fused)
  const kernelSteps: KernelStepSchema[] = [
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
 */
export function isActiveKernelPathDequant(): boolean {
  if (!activeKernelPath) return false;
  const kernelSteps: KernelStepSchema[] = [
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
 */
export function formatKernelPath(path: KernelPathSchema): string {
  const decodeOps = path.decode.steps.map(s => s.op).join(' → ');
  return `${path.id}: ${decodeOps}`;
}

/**
 * Get summary statistics for a kernel path.
 */
export function getKernelPathStats(path: KernelPathSchema): {
  decodeSteps: number;
  prefillSteps: number;
  uniqueKernels: number;
  hasLayerOverrides: boolean;
} {
  const allKernels = new Set<string>();

  const collectKernels = (steps: KernelStepSchema[]) => {
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
