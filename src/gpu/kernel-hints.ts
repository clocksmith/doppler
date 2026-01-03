/**
 * Kernel Hints Configuration
 *
 * Stores kernel selection hints from the manifest or runtime overrides.
 * These hints guide kernel selection in matmul.ts and other kernel modules.
 *
 * Flow:
 * 1. Pipeline loads manifest with optimizations.kernelHints
 * 2. Pipeline calls setKernelHints() to configure globally
 * 3. matmul.ts calls getKernelHints() to check before selecting kernel
 *
 * Override priority (highest to lowest):
 * 1. Runtime API (setKernelHints with override=true)
 * 2. YAML profile (future)
 * 3. Manifest defaults (optimizations.kernelHints)
 * 4. Built-in heuristics (when no hints provided)
 */

import type { KernelHints } from '../storage/rdrr-format.js';
import { log } from '../debug/index.js';

// Module-level state
let currentHints: KernelHints | null = null;
let hintsSource: 'manifest' | 'profile' | 'runtime' | null = null;

/**
 * Set kernel hints from manifest or runtime override.
 *
 * Higher priority sources MERGE with existing hints, not replace.
 * This allows runtime to override specific hints while keeping defaults.
 */
export function setKernelHints(hints: KernelHints, source: 'manifest' | 'profile' | 'runtime' = 'manifest'): void {
  // Runtime overrides everything, profile overrides manifest
  const priority = { manifest: 0, profile: 1, runtime: 2 };
  const currentPriority = priority[hintsSource || 'manifest'];
  const newPriority = priority[source];

  if (!currentHints) {
    // First hints - just set them
    currentHints = { ...hints };
    hintsSource = source;
    log.debug('KernelHints', `Set from ${source}: ${JSON.stringify(currentHints)}`);
  } else if (newPriority > currentPriority) {
    // Higher priority source - MERGE new hints over existing
    // This lets runtime override specific fields while keeping defaults
    currentHints = { ...currentHints, ...hints };
    hintsSource = source;
    log.debug('KernelHints', `Merged from ${source}: ${JSON.stringify(hints)} -> ${JSON.stringify(currentHints)}`);
  } else if (newPriority === currentPriority) {
    // Same priority - merge (later call wins per-field)
    currentHints = { ...currentHints, ...hints };
    log.debug('KernelHints', `Updated from ${source}: ${JSON.stringify(currentHints)}`);
  }
  // Lower priority hints are ignored
}

/**
 * Get current kernel hints.
 * Returns null if no hints have been set.
 */
export function getKernelHints(): KernelHints | null {
  return currentHints;
}

/**
 * Get the source of current hints.
 */
export function getKernelHintsSource(): string | null {
  return hintsSource;
}

/**
 * Clear kernel hints (for testing or model unload).
 */
export function clearKernelHints(): void {
  currentHints = null;
  hintsSource = null;
}

/**
 * Check if Q4K should use fused kernel or dequant path.
 * Based on hint value or falls back to manifest q4kMatmul hint.
 *
 * Default: true (use fused) - keeps ~5GB Q4K weights in VRAM instead of ~20GB F16.
 * Trade-off: fused is slower per-token but fits large models in memory.
 * For maximum speed on models that fit dequantized, set q4kMatmul: 'dequant_f16'.
 */
export function shouldUseFusedQ4K(): boolean {
  // Check window override first (debug flag)
  const debugFlags = typeof window !== 'undefined'
    ? (window as unknown as { DOPPLER_DISABLE_FUSED_Q4K?: boolean })
    : null;
  if (debugFlags?.DOPPLER_DISABLE_FUSED_Q4K) return false;

  // Check kernel hints
  const hints = getKernelHints();
  if (hints?.q4kMatmul) {
    // 'fused_q4k' means use fused, anything else (like 'dequant_f16') means don't
    return hints.q4kMatmul === 'fused_q4k';
  }

  // Default: use fused to keep Q4K weights compressed in VRAM (4x memory savings)
  // This matches the loader's default behavior when subgroups are available.
  // Trade-off: ~2x slower inference, but large models (9B+) actually fit in RAM.
  return true;
}

/**
 * Get preferred compute precision.
 * - 'f16': Fast F16 arithmetic (requires shader-f16)
 * - 'f32': Compatible F32 arithmetic
 * - 'auto': Detect at runtime (default)
 */
export function getComputePrecision(): 'f16' | 'f32' | 'auto' {
  const hints = getKernelHints();
  return hints?.computePrecision || 'auto';
}

/**
 * Check if F16 compute should be used based on hints and GPU capabilities.
 * @param hasShaderF16 - Whether the GPU supports shader-f16
 */
export function shouldUseF16Compute(hasShaderF16: boolean): boolean {
  const precision = getComputePrecision();

  if (precision === 'f16') {
    if (!hasShaderF16) {
      log.warn('KernelHints', 'F16 compute requested but shader-f16 not available, falling back to F32');
      return false;
    }
    return true;
  }

  if (precision === 'f32') {
    return false;
  }

  // auto: use F16 if available
  return hasShaderF16;
}
