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

import { getMemoryCapabilities, type MemoryCapabilities } from '../memory/capability.js';
import { getQuotaInfo, formatBytes, type QuotaInfo } from './quota.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * VRAM check result
 */
export interface VRAMCheckResult {
  /** Bytes required for inference */
  required: number;
  /** Estimated available VRAM in bytes */
  available: number;
  /** True if available >= required */
  sufficient: boolean;
  /** Human-readable message */
  message: string;
}

/**
 * Storage check result
 */
export interface StorageCheckResult {
  /** Download size in bytes */
  required: number;
  /** Available OPFS space in bytes */
  available: number;
  /** True if available >= required */
  sufficient: boolean;
  /** Human-readable message */
  message: string;
}

/**
 * GPU info result
 */
export interface GPUCheckResult {
  /** WebGPU is available */
  hasWebGPU: boolean;
  /** shader-f16 feature available */
  hasF16: boolean;
  /** Device description */
  device: string;
  /** Is unified memory (Apple/AMD) */
  isUnified: boolean;
}

/**
 * Complete pre-flight check result
 */
export interface PreflightResult {
  /** Overall: can proceed with download */
  canProceed: boolean;
  /** VRAM check details */
  vram: VRAMCheckResult;
  /** Storage check details */
  storage: StorageCheckResult;
  /** GPU capability details */
  gpu: GPUCheckResult;
  /** Warning messages (non-blocking) */
  warnings: string[];
  /** Blocker messages (prevents download) */
  blockers: string[];
}

/**
 * Model requirements definition
 */
export interface ModelRequirements {
  /** Model identifier */
  modelId: string;
  /** Display name */
  displayName: string;
  /** Total download size in bytes */
  downloadSize: number;
  /** VRAM required for inference in bytes */
  vramRequired: number;
  /** Parameter count string (e.g., "1B", "7B") */
  paramCount: string;
  /** Quantization type (e.g., "Q4_K_M", "BF16") */
  quantization: string;
  /** Model architecture (e.g., "gemma3", "llama") */
  architecture?: string;
}

// ============================================================================
// Model Requirements Constants
// ============================================================================

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;

/**
 * Gemma 1B requirements (Q4_K_M quantization)
 */
export const GEMMA_1B_REQUIREMENTS: ModelRequirements = {
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
 */
export const MODEL_REQUIREMENTS: Record<string, ModelRequirements> = {
  'gemma-3-1b-it-q4': GEMMA_1B_REQUIREMENTS,
};

// ============================================================================
// VRAM Estimation
// ============================================================================

/**
 * Estimate available VRAM based on memory capabilities
 */
function estimateAvailableVRAM(memCaps: MemoryCapabilities): number {
  const info = memCaps.unifiedMemoryInfo;

  // Unified memory: use estimated system memory (leave headroom)
  if (info.isUnified && info.estimatedMemoryGB) {
    // Assume 50% of unified memory available for GPU
    // (rest is used by OS, apps, etc.)
    return (info.estimatedMemoryGB * GB) * 0.5;
  }

  // Discrete GPU: use maxBufferSize as heuristic
  // This isn't actual VRAM but gives us a sense of GPU capability
  if (info.limits?.maxBufferSize) {
    // Conservative: discrete GPUs can usually allocate ~80% of VRAM
    return info.limits.maxBufferSize;
  }

  // Fallback: assume 2GB (conservative for most GPUs)
  return 2 * GB;
}

/**
 * Check VRAM sufficiency
 */
async function checkVRAM(
  requirements: ModelRequirements,
  memCaps: MemoryCapabilities
): Promise<VRAMCheckResult> {
  const available = estimateAvailableVRAM(memCaps);
  const required = requirements.vramRequired;
  const sufficient = available >= required;

  let message: string;
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
 */
async function checkStorage(
  requirements: ModelRequirements
): Promise<StorageCheckResult> {
  const quotaInfo = await getQuotaInfo();
  const available = quotaInfo.available;
  const required = requirements.downloadSize;
  const sufficient = available >= required;

  let message: string;
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
 */
async function checkGPU(memCaps: MemoryCapabilities): Promise<GPUCheckResult> {
  const hasWebGPU = !!navigator.gpu;

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
 * @param requirements - Model requirements to check against
 * @returns Pre-flight check result with all details
 *
 * @example
 * ```typescript
 * const result = await runPreflightChecks(GEMMA_1B_REQUIREMENTS);
 * if (!result.canProceed) {
 *   console.error('Cannot download:', result.blockers.join(', '));
 * }
 * ```
 */
export async function runPreflightChecks(
  requirements: ModelRequirements
): Promise<PreflightResult> {
  const warnings: string[] = [];
  const blockers: string[] = [];

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
    if (headroom < 500 * MB) {
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
 */
export function formatPreflightResult(result: PreflightResult): string {
  const lines: string[] = [];

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
