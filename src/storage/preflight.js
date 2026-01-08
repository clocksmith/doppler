/**
 * preflight.ts - Pre-download Validation
 *
 * Performs checks before model download:
 * - VRAM estimation and validation
 * - Storage space availability
 * - GPU capability verification
 *
 * @module storage/preflight
 */

import { getMemoryCapabilities } from '../memory/capability.js';
import { getQuotaInfo, formatBytes } from './quota.js';
import { getRuntimeConfig } from '../config/runtime.js';

// ============================================================================
// Model Requirements Constants
// ============================================================================

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;

/**
 * Gemma 1B requirements (Q4_K_M quantization)
 * @type {import('./preflight.js').ModelRequirements}
 */
export const GEMMA_1B_REQUIREMENTS = {
  modelId: 'gemma-3-1b-it-q4',
  displayName: 'Gemma 3 1B IT (Q4)',
  downloadSize: 537 * MB,       // ~537MB actual size from manifest shards
  vramRequired: 1.5 * GB,       // ~1.5GB VRAM for inference (weights + KV cache)
  paramCount: '1B',
  quantization: 'Q4_K_M',
  architecture: 'Gemma3ForCausalLM',
};

/**
 * All available model requirements
 * Naming convention: {family}-{version}-{size}-{variant}-{quant}
 * Examples: gemma-3-1b-it-q4, llama-3.2-1b-q4, mistral-7b-v0.3-q4
 * @type {Record<string, import('./preflight.js').ModelRequirements>}
 */
export const MODEL_REQUIREMENTS = {
  'gemma-3-1b-it-q4': GEMMA_1B_REQUIREMENTS,
};

// ============================================================================
// VRAM Estimation
// ============================================================================

/**
 * Estimate available VRAM based on memory capabilities
 * @param {import('../memory/capability.js').MemoryCapabilities} memCaps
 * @returns {number}
 */
function estimateAvailableVRAM(memCaps) {
  const info = memCaps.unifiedMemoryInfo;

  // Unified memory: use estimated system memory (leave headroom)
  if (info.isUnified && info.estimatedMemoryGB) {
    // Use configured ratio of unified memory available for GPU
    return (info.estimatedMemoryGB * GB) * getRuntimeConfig().storage.vramEstimation.unifiedMemoryRatio;
  }

  // Discrete GPU: use maxBufferSize as heuristic
  // This isn't actual VRAM but gives us a sense of GPU capability
  if (info.limits?.maxBufferSize) {
    // Conservative: discrete GPUs can usually allocate ~80% of VRAM
    return info.limits.maxBufferSize;
  }

  // Fallback: use configured fallback VRAM
  return getRuntimeConfig().storage.vramEstimation.fallbackVramBytes;
}

/**
 * Check VRAM sufficiency
 * @param {import('./preflight.js').ModelRequirements} requirements
 * @param {import('../memory/capability.js').MemoryCapabilities} memCaps
 * @returns {Promise<import('./preflight.js').VRAMCheckResult>}
 */
async function checkVRAM(
  requirements,
  memCaps
) {
  const available = estimateAvailableVRAM(memCaps);
  const required = requirements.vramRequired;
  const sufficient = available >= required;

  /** @type {string} */
  let message;
  if (sufficient) {
    message = `VRAM OK: ${formatBytes(available)} available, ${formatBytes(required)} required`;
  } else {
    message = `Insufficient VRAM: ${formatBytes(available)} available, ${formatBytes(required)} required`;
  }

  return { required, available, sufficient, message };
}

// ============================================================================
// Storage Check
// ============================================================================

/**
 * Check storage space sufficiency
 * @param {import('./preflight.js').ModelRequirements} requirements
 * @returns {Promise<import('./preflight.js').StorageCheckResult>}
 */
async function checkStorage(
  requirements
) {
  const quotaInfo = await getQuotaInfo();
  const available = quotaInfo.available;
  const required = requirements.downloadSize;
  const sufficient = available >= required;

  /** @type {string} */
  let message;
  if (sufficient) {
    message = `Storage OK: ${formatBytes(available)} available, ${formatBytes(required)} required`;
  } else {
    const shortfall = required - available;
    message = `Insufficient storage: need ${formatBytes(shortfall)} more space`;
  }

  return { required, available, sufficient, message };
}

// ============================================================================
// GPU Check
// ============================================================================

/**
 * Check GPU capabilities
 * @param {import('../memory/capability.js').MemoryCapabilities} memCaps
 * @returns {Promise<import('./preflight.js').GPUCheckResult>}
 */
async function checkGPU(memCaps) {
  const hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;

  if (!hasWebGPU) {
    return {
      hasWebGPU: false,
      hasF16: false,
      device: 'WebGPU not available',
      isUnified: false,
    };
  }

  const info = memCaps.unifiedMemoryInfo;
  let device = 'Unknown GPU';

  if (info.apple?.isApple) {
    device = info.apple.description || `Apple M${info.apple.mSeriesGen || '?'}`;
  } else if (info.amd?.isAMDUnified) {
    device = info.amd.description || 'AMD Strix';
  } else if (info.limits?.maxBufferSize) {
    device = `GPU (${formatBytes(info.limits.maxBufferSize)} max buffer)`;
  }

  // Check for F16 support (need to request adapter to check features)
  let hasF16 = false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) {
      hasF16 = adapter.features.has('shader-f16');
    }
  } catch {
    // Ignore - hasF16 stays false
  }

  return {
    hasWebGPU: true,
    hasF16,
    device,
    isUnified: info.isUnified,
  };
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Run all pre-flight checks before model download
 *
 * @param {import('./preflight.js').ModelRequirements} requirements - Model requirements to check against
 * @returns {Promise<import('./preflight.js').PreflightResult>} Pre-flight check result with all details
 *
 * @example
 * ```typescript
 * import { log } from '../debug/index.js';
 *
 * const result = await runPreflightChecks(GEMMA_1B_REQUIREMENTS);
 * if (!result.canProceed) {
 *   log.error('Preflight', 'Cannot download', result.blockers);
 * }
 * ```
 */
export async function runPreflightChecks(
  requirements
) {
  /** @type {string[]} */
  const warnings = [];
  /** @type {string[]} */
  const blockers = [];

  // Get memory capabilities (cached internally)
  const memCaps = await getMemoryCapabilities();

  // Run all checks
  const [vram, storage, gpu] = await Promise.all([
    checkVRAM(requirements, memCaps),
    checkStorage(requirements),
    checkGPU(memCaps),
  ]);

  // Determine blockers
  if (!gpu.hasWebGPU) {
    blockers.push('WebGPU is not available in this browser');
  }

  if (!vram.sufficient) {
    blockers.push(vram.message);
  }

  if (!storage.sufficient) {
    blockers.push(storage.message);
  }

  // Determine warnings
  if (!gpu.hasF16) {
    warnings.push('F16 not supported - inference may be slower');
  }

  if (!gpu.isUnified && vram.sufficient) {
    // Discrete GPU with borderline VRAM
    const headroom = vram.available - vram.required;
    if (headroom < getRuntimeConfig().storage.vramEstimation.lowVramHeadroomBytes) {
      warnings.push('Low VRAM headroom - may cause issues with longer contexts');
    }
  }

  const canProceed = blockers.length === 0;

  return {
    canProceed,
    vram,
    storage,
    gpu,
    warnings,
    blockers,
  };
}

/**
 * Format pre-flight result for display
 * @param {import('./preflight.js').PreflightResult} result
 * @returns {string}
 */
export function formatPreflightResult(result) {
  /** @type {string[]} */
  const lines = [];

  lines.push(`GPU: ${result.gpu.device}`);
  lines.push(`VRAM: ${result.vram.message}`);
  lines.push(`Storage: ${result.storage.message}`);

  if (result.warnings.length > 0) {
    lines.push(`Warnings: ${result.warnings.join('; ')}`);
  }

  if (result.blockers.length > 0) {
    lines.push(`Blockers: ${result.blockers.join('; ')}`);
  }

  lines.push(`Can proceed: ${result.canProceed ? 'Yes' : 'No'}`);

  return lines.join('\n');
}
