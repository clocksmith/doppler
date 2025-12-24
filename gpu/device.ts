/**
 * WebGPU Device Initialization and Feature Probing
 *
 * Handles WebGPU adapter/device setup with comprehensive feature detection
 * for optimal kernel selection.
 */

import type { GpuCapabilities, GpuLimits } from '../types/gpu.js';
import { wrapQueueForTracking, TRACK_SUBMITS, setTrackSubmits } from './submit-tracker.js';

// Re-export types for convenience
export type { GpuCapabilities, GpuLimits };

// Re-export submit tracker for convenience
export { setTrackSubmits };

/**
 * GPU adapter information
 */
export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

/**
 * Extended kernel capabilities with adapter info
 */
export interface KernelCapabilities {
  hasSubgroups: boolean;
  hasSubgroupsF16: boolean;
  hasF16: boolean;
  hasTimestampQuery: boolean;
  maxBufferSize: number;
  maxWorkgroupSize: number;
  maxWorkgroupStorageSize: number;
  adapterInfo: AdapterInfo;
  features?: string[];  // GPU feature strings from adapter
}

/**
 * WebGPU adapter request options
 */
interface AdapterRequestOptions {
  powerPreference?: 'low-power' | 'high-performance';
  forceFallbackAdapter?: boolean;
}

/**
 * GPU device limits for initialization
 */
export interface DeviceLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupStorageSize: number;
  maxStorageBuffersPerShaderStage: number;
  maxUniformBufferBindingSize: number;
  maxComputeWorkgroupsPerDimension: number;
}

// Cached device and capabilities
let gpuDevice: GPUDevice | null = null;
let kernelCapabilities: KernelCapabilities | null = null;

/**
 * Feature flags detected during initialization
 */
const FEATURES = {
  SHADER_F16: 'shader-f16',
  SUBGROUPS: 'subgroups',
  SUBGROUPS_F16: 'subgroups-f16',
  TIMESTAMP_QUERY: 'timestamp-query',
} as const;

type FeatureKey = keyof typeof FEATURES;
type FeatureValue = typeof FEATURES[FeatureKey];

/**
 * Probe for WebGPU availability
 */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Request WebGPU adapter with fallback options
 * @param options - Adapter request options
 * @returns GPU adapter or null if unavailable
 */
async function requestAdapter(options: AdapterRequestOptions = {}): Promise<GPUAdapter | null> {
  if (!isWebGPUAvailable()) {
    return null;
  }

  // Try high-performance first, then fallback
  const adapterOptions: GPURequestAdapterOptions[] = [
    { powerPreference: 'high-performance', ...options },
    { powerPreference: 'low-power', ...options },
    { ...options }, // Default
  ];

  for (const opts of adapterOptions) {
    try {
      const adapter = await navigator.gpu.requestAdapter(opts);
      if (adapter) {
        return adapter;
      }
    } catch (e) {
      // Continue to next option
    }
  }

  return null;
}

/**
 * Detect available features from adapter
 * @param adapter - GPU adapter
 * @returns Set of available feature names
 */
function detectFeatures(adapter: GPUAdapter): Set<string> {
  const available = new Set<string>();

  for (const feature of adapter.features) {
    available.add(feature);
  }

  return available;
}

/**
 * Build list of features to request based on availability
 * @param available - Available features
 * @returns Array of feature names to request
 */
function buildFeatureRequests(available: Set<string>): GPUFeatureName[] {
  const requested: GPUFeatureName[] = [];

  // Request shader-f16 for FP16 matmul kernels
  if (available.has(FEATURES.SHADER_F16)) {
    requested.push(FEATURES.SHADER_F16 as GPUFeatureName);
  }

  // Request subgroups for efficient dequantization
  if (available.has(FEATURES.SUBGROUPS)) {
    requested.push(FEATURES.SUBGROUPS as GPUFeatureName);
  }

  // Request subgroups-f16 if available (for combined f16 + subgroup ops)
  if (available.has(FEATURES.SUBGROUPS_F16)) {
    requested.push(FEATURES.SUBGROUPS_F16 as GPUFeatureName);
  }

  // Request timestamp query for profiling (optional)
  if (available.has(FEATURES.TIMESTAMP_QUERY)) {
    requested.push(FEATURES.TIMESTAMP_QUERY as GPUFeatureName);
  }

  return requested;
}

/**
 * Build device limits based on adapter capabilities
 * @param adapter - GPU adapter
 * @returns Device limits object
 */
function buildLimits(adapter: GPUAdapter): Record<string, GPUSize64> {
  const adapterLimits = adapter.limits;

  return {
    // Request maximum available storage buffer size (critical for large weight tensors)
    maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
    // Request maximum buffer size
    maxBufferSize: adapterLimits.maxBufferSize,
    // Request maximum compute workgroup sizes
    maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: adapterLimits.maxComputeWorkgroupStorageSize,
    // Binding limits
    maxStorageBuffersPerShaderStage: adapterLimits.maxStorageBuffersPerShaderStage,
    maxUniformBufferBindingSize: adapterLimits.maxUniformBufferBindingSize,
  };
}

/**
 * Initialize WebGPU device with optimal features
 * @returns GPU device
 * @throws Error if WebGPU is unavailable or device creation fails
 */
export async function initDevice(): Promise<GPUDevice> {
  // Return cached device if available
  if (gpuDevice) {
    return gpuDevice;
  }

  if (!isWebGPUAvailable()) {
    throw new Error('WebGPU is not available in this browser');
  }

  const adapter = await requestAdapter();
  if (!adapter) {
    throw new Error('Failed to get WebGPU adapter');
  }

  // Detect available features
  const availableFeatures = detectFeatures(adapter);
  const requestedFeatures = buildFeatureRequests(availableFeatures);
  const limits = buildLimits(adapter);

  // Get adapter info (adapter.info is synchronous in modern WebGPU)
  const adapterInfo = adapter.info || { vendor: 'unknown', architecture: 'unknown', device: 'unknown', description: '' };

  try {
    gpuDevice = await adapter.requestDevice({
      requiredFeatures: requestedFeatures,
      requiredLimits: limits,
    });
  } catch (e) {
    // Fallback: request device without optional features
    console.warn('[DOPPLER GPU] Failed to request device with features, trying minimal config:', (e as Error).message);
    gpuDevice = await adapter.requestDevice();
  }

  if (!gpuDevice) {
    throw new Error('Failed to create WebGPU device');
  }

  // Set up device lost handler
  gpuDevice.lost.then((info) => {
    console.error('[DOPPLER GPU] Device lost:', info.message, 'Reason:', info.reason);
    gpuDevice = null;
    kernelCapabilities = null;
  });

  // Wrap queue for submit tracking (when enabled)
  wrapQueueForTracking(gpuDevice.queue);

  // Cache kernel capabilities
  kernelCapabilities = {
    hasSubgroups: gpuDevice.features.has(FEATURES.SUBGROUPS),
    hasSubgroupsF16: gpuDevice.features.has(FEATURES.SUBGROUPS_F16),
    hasF16: gpuDevice.features.has(FEATURES.SHADER_F16),
    hasTimestampQuery: gpuDevice.features.has(FEATURES.TIMESTAMP_QUERY),
    maxBufferSize: gpuDevice.limits.maxStorageBufferBindingSize,
    maxWorkgroupSize: gpuDevice.limits.maxComputeInvocationsPerWorkgroup,
    maxWorkgroupStorageSize: gpuDevice.limits.maxComputeWorkgroupStorageSize,
    adapterInfo: {
      vendor: adapterInfo.vendor || 'unknown',
      architecture: adapterInfo.architecture || 'unknown',
      device: adapterInfo.device || 'unknown',
      description: adapterInfo.description || '',
    },
  };

  const features = [
    kernelCapabilities.hasF16 && 'f16',
    kernelCapabilities.hasSubgroups && 'subgroups',
  ].filter(Boolean).join('/') || 'basic';
  console.log(`[GPU] ${adapterInfo.vendor || 'unknown'} ${adapterInfo.architecture || adapterInfo.device || ''}, ${features}, ${(kernelCapabilities.maxBufferSize / (1024 * 1024 * 1024)).toFixed(1)}GB`);

  return gpuDevice;
}

/**
 * Get kernel capabilities (call after initDevice)
 * @returns Capability flags for kernel selection
 * @throws Error if device not initialized
 */
export function getKernelCapabilities(): KernelCapabilities {
  if (!kernelCapabilities) {
    throw new Error('Device not initialized. Call initDevice() first.');
  }
  return { ...kernelCapabilities };
}

/**
 * Get the current GPU device (call after initDevice)
 * @returns GPU device or null if not initialized
 */
export function getDevice(): GPUDevice | null {
  return gpuDevice;
}

/**
 * Destroy the device and clear cache
 */
export function destroyDevice(): void {
  if (gpuDevice) {
    gpuDevice.destroy();
    gpuDevice = null;
    kernelCapabilities = null;
  }
}

/**
 * Check if a specific feature is available
 * @param feature - Feature name to check
 * @returns True if feature is available
 */
export function hasFeature(feature: string): boolean {
  if (!gpuDevice) {
    return false;
  }
  return gpuDevice.features.has(feature as GPUFeatureName);
}

/**
 * Get device limits
 * @returns Device limits or null if device not initialized
 */
export function getDeviceLimits(): DeviceLimits | null {
  if (!gpuDevice) {
    return null;
  }
  return {
    maxStorageBufferBindingSize: gpuDevice.limits.maxStorageBufferBindingSize,
    maxBufferSize: gpuDevice.limits.maxBufferSize,
    maxComputeWorkgroupSizeX: gpuDevice.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: gpuDevice.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: gpuDevice.limits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: gpuDevice.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: gpuDevice.limits.maxComputeWorkgroupStorageSize,
    maxStorageBuffersPerShaderStage: gpuDevice.limits.maxStorageBuffersPerShaderStage,
    maxUniformBufferBindingSize: gpuDevice.limits.maxUniformBufferBindingSize,
    maxComputeWorkgroupsPerDimension: gpuDevice.limits.maxComputeWorkgroupsPerDimension,
  };
}

// Feature constants for external use
export { FEATURES };
