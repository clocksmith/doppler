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

export interface StorageBackendCheckResult {
  requested: string;
  selected: string;
  persistent: boolean;
  ok: boolean;
  message: string;
  caps: {
    opfs: boolean;
    indexeddb: boolean;
    sharedArrayBuffer: boolean;
    byob: boolean;
    syncAccessHandle: boolean;
  };
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
  /** Storage backend selection and capabilities */
  storageBackend: StorageBackendCheckResult;
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

/**
 * Gemma 1B requirements (Q4_K_M quantization)
 */
export declare const GEMMA_1B_REQUIREMENTS: ModelRequirements;

/**
 * All available model requirements
 * Naming convention: {family}-{version}-{size}-{variant}-{quant}
 * Examples: gemma-3-1b-it-wq4k, llama-3.2-1b-q4, mistral-7b-v0.3-q4
 */
export declare const MODEL_REQUIREMENTS: Record<string, ModelRequirements>;

/**
 * Run all pre-flight checks before model download
 *
 * @param requirements - Model requirements to check against
 * @returns Pre-flight check result with all details
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
export declare function runPreflightChecks(
  requirements: ModelRequirements
): Promise<PreflightResult>;

/**
 * Format pre-flight result for display
 */
export declare function formatPreflightResult(result: PreflightResult): string;
