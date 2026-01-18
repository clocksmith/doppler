/**
 * Shared utilities for virtual device layer.
 * @module simulator/virtual-utils
 */

/** Module name for logging */
export const MODULE = 'VirtualDevice';

/** Maximum bytes to keep in actual VRAM per virtual GPU */
export const DEFAULT_VRAM_BUDGET_BYTES = 2 * 1024 * 1024 * 1024; // 2GB

/** Buffer ID counter */
let bufferIdCounter = 0;

/**
 * Generate a unique buffer ID
 * @returns {string} Unique buffer ID
 */
export function generateBufferId() {
  return `vbuf_${Date.now()}_${bufferIdCounter++}`;
}
