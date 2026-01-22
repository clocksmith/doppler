import { DEFAULT_ENTRY } from './schema/kernel-path.schema.js';
import { getKernelConfig } from '../gpu/kernels/utils.js';
import { log } from '../debug/index.js';

// =============================================================================
// Built-in Kernel Paths (imported at build time)
// =============================================================================

import gemma2Q4kFusedF16A from './presets/kernel-paths/gemma2-q4k-fused-f16a.json' with { type: 'json' };
import gemma2Q4kFusedF32A from './presets/kernel-paths/gemma2-q4k-fused-f32a.json' with { type: 'json' };
import gemma2Q4kDequantF16A from './presets/kernel-paths/gemma2-q4k-dequant-f16a.json' with { type: 'json' };
import gemma2F16F16A from './presets/kernel-paths/gemma2-f16-f16a.json' with { type: 'json' };
import gemma2F16F32A from './presets/kernel-paths/gemma2-f16-f32a.json' with { type: 'json' };
import gemma3F16F16A from './presets/kernel-paths/gemma3-f16-f16a.json' with { type: 'json' };
import gemma3F16F16AOnline from './presets/kernel-paths/gemma3-f16-f16a-online.json' with { type: 'json' };
import gemma3Q4kFusedF16A from './presets/kernel-paths/gemma3-q4k-fused-f16a.json' with { type: 'json' };
import gemma3Q4kDequantF16A from './presets/kernel-paths/gemma3-q4k-dequant-f16a.json' with { type: 'json' };

const KERNEL_PATH_REGISTRY = {
  // Gemma 2 Q4K variants
  'gemma2-q4k-fused-f16a': gemma2Q4kFusedF16A,
  'gemma2-q4k-fused-f32a': gemma2Q4kFusedF32A,
  'gemma2-q4k-dequant-f16a': gemma2Q4kDequantF16A,

  // Gemma 2 F16 variants
  'gemma2-f16-f16a': gemma2F16F16A,
  'gemma2-f16-f32a': gemma2F16F32A,

  // Gemma 3 variants
  'gemma3-f16-f16a': gemma3F16F16A,
  'gemma3-f16-f16a-online': gemma3F16F16AOnline,
  'gemma3-q4k-fused-f16a': gemma3Q4kFusedF16A,
  'gemma3-q4k-dequant-f16a': gemma3Q4kDequantF16A,
};

// =============================================================================
// Public API
// =============================================================================

export function getKernelPath(id) {
  return KERNEL_PATH_REGISTRY[id] ?? null;
}

export function listKernelPaths() {
  return Object.keys(KERNEL_PATH_REGISTRY);
}

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

export function getKernelPathActivationDtype(path) {
  if (!path?.activationDtype) return null;
  return path.activationDtype;
}

export function applyKernelOverrides(path, overrides) {
  if (!overrides) return path;

  const cloned = structuredClone(path);

  const resolveOverrideConfig = (variantId, opCandidates) => {
    const ops = Array.isArray(opCandidates) ? opCandidates : [opCandidates];
    for (const op of ops) {
      try {
        return getKernelConfig(op, variantId);
      } catch {
        continue;
      }
    }
    return null;
  };

  const applyToStep = (steps, stepOp, variantId, opCandidates) => {
    if (!steps || variantId == null) return;
    const step = steps.find((s) => s.op === stepOp);
    if (!step) return;
    const config = resolveOverrideConfig(variantId, opCandidates);
    if (!config) {
      log.warn('KernelOverrides', `Variant '${variantId}' not found for op '${stepOp}'.`);
      return;
    }
    step.kernel = config.shaderFile;
    step.entry = config.entryPoint;
  };

  // 1. Attention Overrides
  if (overrides.attention) {
    if (overrides.attention.decode) {
      applyToStep(cloned.decode?.steps, 'attention', overrides.attention.decode, 'attention');
    }
    if (overrides.attention.prefill) {
      applyToStep(cloned.prefill?.steps, 'attention', overrides.attention.prefill, 'attention');
    }
  }

  // 2. Matmul Overrides (Layer & Head)
  if (overrides.matmul) {
    const matmulOps = [
      'q_proj', 'k_proj', 'v_proj', 'o_proj',
      'gate_proj', 'up_proj', 'down_proj'
    ];
    
    // Apply to both decode and prefill phases for layer weights
    for (const op of matmulOps) {
      if (overrides.matmul[op]) {
        applyToStep(cloned.decode?.steps, op, overrides.matmul[op], 'matmul');
        applyToStep(cloned.prefill?.steps, op, overrides.matmul[op], 'matmul');
      }
    }

    // LM Head (Post-layer)
    if (overrides.matmul.lm_head) {
      applyToStep(cloned.postLayer, 'lm_head', overrides.matmul.lm_head, 'matmul');
      applyToStep(cloned.postLayer, 'lm_head_prefill', overrides.matmul.lm_head, 'matmul');
    }
  }

  // 3. FFN Overrides
  if (overrides.ffn) {
    if (overrides.ffn.activation) {
      applyToStep(cloned.decode?.steps, 'activation', overrides.ffn.activation, ['gelu', 'silu']);
      applyToStep(cloned.prefill?.steps, 'activation', overrides.ffn.activation, ['gelu', 'silu']);
    }
    if (overrides.ffn.rmsnorm) {
      // Maps to post_attn_norm (the FFN input norm)
      applyToStep(cloned.decode?.steps, 'post_attn_norm', overrides.ffn.rmsnorm, 'rmsnorm');
      applyToStep(cloned.prefill?.steps, 'post_attn_norm', overrides.ffn.rmsnorm, 'rmsnorm');
    }
  }

  // 4. RoPE Overrides
  if (overrides.rope) {
    if (overrides.rope.q) {
      applyToStep(cloned.decode?.steps, 'rope_q', overrides.rope.q, 'rope');
      applyToStep(cloned.prefill?.steps, 'rope_q', overrides.rope.q, 'rope');
    }
    if (overrides.rope.k) {
      applyToStep(cloned.decode?.steps, 'rope_k', overrides.rope.k, 'rope');
      applyToStep(cloned.prefill?.steps, 'rope_k', overrides.rope.k, 'rope');
    }
  }

  // 5. Residual Overrides
  if (overrides.residual) {
    if (overrides.residual.attn) {
      applyToStep(cloned.decode?.steps, 'attn_residual', overrides.residual.attn, 'residual');
      applyToStep(cloned.prefill?.steps, 'attn_residual', overrides.residual.attn, 'residual');
    }
    if (overrides.residual.ffn) {
      applyToStep(cloned.decode?.steps, 'ffn_residual', overrides.residual.ffn, 'residual');
      applyToStep(cloned.prefill?.steps, 'ffn_residual', overrides.residual.ffn, 'residual');
    }
  }

  if (overrides.kv?.quantize) {
    applyToStep(cloned.decode?.steps, 'kv_quantize', overrides.kv.quantize, 'kv_quantize');
    applyToStep(cloned.prefill?.steps, 'kv_quantize', overrides.kv.quantize, 'kv_quantize');
  }

  return cloned;
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

function findKernelVariant(
  operation,
  kernel,
  entry,
  phase
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
      entryMatches.push(variant);
    }
  }

  if (entryMatches.length === 1) {
    return entryMatches[0];
  }
  if (entryMatches.length > 1 && phase) {
    const phasePrefix = `${phase}_`;
    const phaseMatch = entryMatches.find((variant) => variant.startsWith(phasePrefix));
    if (phaseMatch) {
      return phaseMatch;
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
  layerIndex
) {
  const step = getKernelPathMatmulStep(role, phase, layerIndex);
  if (!step) return null;
  return findKernelVariant('matmul', step.kernel, step.entry, phase);
}

export function getKernelPathMatmulConstants(
  role,
  phase,
  layerIndex
) {
  const step = getKernelPathMatmulStep(role, phase, layerIndex);
  return step?.constants ?? null;
}

function getKernelPathMatmulStep(
  role,
  phase,
  layerIndex
) {
  if (!activeKernelPath || !role) return null;
  const alias = MATMUL_ROLE_ALIASES[role] ?? { section: 'layer', ops: [role] };
  const steps = getKernelPathStepsForSection(activeKernelPath, alias.section, phase, layerIndex ?? 0);
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
  layerIndex
) {
  if (!activeKernelPath) return null;
  const steps = getKernelPathStepsForSection(activeKernelPath, 'layer', phase, layerIndex ?? 0);
  const step = findStepByOp(steps, 'attention');
  if (!step) return null;
  return findKernelVariant('attention', step.kernel, step.entry, phase);
}

// =============================================================================
// Active Kernel Path Registry
// =============================================================================

let activeKernelPath = null;
let activeKernelPathSource = 'none';

export function setActiveKernelPath(path, source = 'none') {
  activeKernelPath = path;
  activeKernelPathSource = path ? source : 'none';
}

export function getActiveKernelPath() {
  return activeKernelPath;
}

export function getActiveKernelPathSource() {
  return activeKernelPathSource;
}

export function getKernelPathStrict() {
  return true;
}

export function isActiveKernelPathFusedQ4K() {
  if (!activeKernelPath) return true; // Default to fused when no explicit path is set
  const kernelSteps = [
    ...(activeKernelPath.decode?.steps ?? []),
    ...(activeKernelPath.prefill?.steps ?? []),
    ...(activeKernelPath.preLayer ?? []),
    ...(activeKernelPath.postLayer ?? []),
    ...(activeKernelPath.layerOverrides?.flatMap((override) => override.steps) ?? []),
  ];
  return kernelSteps.some((step) => step.kernel.includes('fused_matmul_q4'));
}

export function isActiveKernelPathDequant() {
  if (!activeKernelPath) return false;
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

export function formatKernelPath(path) {
  const decodeOps = path.decode.steps.map(s => s.op).join(' → ');
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
