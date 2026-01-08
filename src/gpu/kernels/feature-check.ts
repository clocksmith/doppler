/**
 * Feature Check - GPU capability checking utilities
 *
 * Provides utilities for checking device feature requirements.
 *
 * @module gpu/kernels/feature-check
 */

import { getDeviceLimits } from '../device.js';
import { log } from '../../debug/index.js';

// ============================================================================
// Types
// ============================================================================

/** Minimum capabilities interface for feature checking */
export interface FeatureCapabilities {
  hasF16: boolean;
  hasSubgroups: boolean;
}

// ============================================================================
// Feature Checking
// ============================================================================

/**
 * Check if all required features are available
 */
export function hasRequiredFeatures(
  required: string[],
  capabilities: FeatureCapabilities
): boolean {
  for (const feature of required) {
    if (feature === 'shader-f16' && !capabilities.hasF16) return false;
    if (feature === 'subgroups' && !capabilities.hasSubgroups) return false;
    if (feature === 'subgroups-f16' && !capabilities.hasSubgroups) return false;
  }
  return true;
}

// ============================================================================
// Attention Validation
// ============================================================================

/**
 * Validate that attention parameters are within device limits
 */
export function validateAttentionLimits(
  seqLen: number,
  numHeads: number,
  headDim: number
): void {
  const limits = getDeviceLimits();
  if (!limits) return; // No device, validation will fail later

  // Check workgroup invocations limit
  const workgroupInvocations = seqLen * numHeads;
  if (workgroupInvocations > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention parameters exceed device limits: ${workgroupInvocations} workgroups ` +
      `> ${limits.maxComputeWorkgroupsPerDimension} max per dimension. ` +
      `Try reducing seqLen (${seqLen}) or numHeads (${numHeads}).`
    );
  }

  // Check buffer size limits for KV cache
  const kvCacheSize = seqLen * numHeads * headDim * 4; // float32
  if (kvCacheSize > limits.maxStorageBufferBindingSize) {
    throw new Error(
      `KV cache size ${(kvCacheSize / 1e9).toFixed(2)}GB exceeds device limit ` +
      `${(limits.maxStorageBufferBindingSize / 1e9).toFixed(2)}GB. ` +
      `Reduce sequence length or use paged attention.`
    );
  }

  // Check shared memory requirements for attention tile
  const tileSize = 64; // TILE_SIZE in attention.wgsl
  const sharedMemRequired = tileSize * headDim * 4 * 2; // K and V tiles
  if (sharedMemRequired > limits.maxComputeWorkgroupStorageSize) {
    log.warn(
      'FeatureCheck',
      `Attention may be slow: tile requires ${sharedMemRequired} bytes but device has ` +
      `${limits.maxComputeWorkgroupStorageSize} bytes shared memory.`
    );
  }
}
