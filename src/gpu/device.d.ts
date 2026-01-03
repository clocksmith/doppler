/**
 * WebGPU Device Initialization and Feature Probing
 *
 * Handles WebGPU adapter/device setup with comprehensive feature detection
 * for optimal kernel selection.
 *
 * Also initializes the platform loader and kernel registry for config-as-code
 * kernel selection based on detected GPU hardware.
 */

import type { GpuCapabilities, GpuLimits } from '../types/gpu.js';
import type { ResolvedPlatformConfig } from '../config/schema/platform.schema.js';

// Re-export types for convenience
export type { GpuCapabilities, GpuLimits };

// Re-export submit tracker for convenience
export { setTrackSubmits } from './submit-tracker.js';

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
  features?: string[];
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

/**
 * Feature flags detected during initialization
 */
export declare const FEATURES: {
  readonly SHADER_F16: 'shader-f16';
  readonly SUBGROUPS: 'subgroups';
  readonly SUBGROUPS_F16: 'subgroups-f16';
  readonly TIMESTAMP_QUERY: 'timestamp-query';
};

/**
 * Probe for WebGPU availability
 */
export function isWebGPUAvailable(): boolean;

/**
 * Initialize WebGPU device with optimal features
 * @returns GPU device
 * @throws Error if WebGPU is unavailable or device creation fails
 */
export function initDevice(): Promise<GPUDevice>;

/**
 * Register an externally created GPU device for pipeline use.
 */
export function setDevice(
  device: GPUDevice | null,
  options?: {
    adapterInfo?: AdapterInfo;
    platformConfig?: ResolvedPlatformConfig | null;
  }
): void;

/**
 * Get kernel capabilities (call after initDevice)
 * @returns Capability flags for kernel selection
 * @throws Error if device not initialized
 */
export function getKernelCapabilities(): KernelCapabilities;

/**
 * Get the current GPU device (call after initDevice)
 * @returns GPU device or null if not initialized
 */
export function getDevice(): GPUDevice | null;

/**
 * Monotonic epoch for device-scoped GPU objects.
 * Increments whenever the active device changes or is lost/destroyed.
 */
export function getDeviceEpoch(): number;

/**
 * Get the resolved platform configuration (call after initDevice)
 * @returns Platform config with capabilities, or null if not initialized or detection failed
 */
export function getPlatformConfig(): ResolvedPlatformConfig | null;

/**
 * Check if platform and registry are initialized
 * @returns True if platform detection succeeded
 */
export function isPlatformInitialized(): boolean;

/**
 * Destroy the device and clear cache
 */
export function destroyDevice(): void;

/**
 * Check if a specific feature is available
 * @param feature - Feature name to check
 * @returns True if feature is available
 */
export function hasFeature(feature: string): boolean;

/**
 * Get device limits
 * @returns Device limits or null if device not initialized
 */
export function getDeviceLimits(): DeviceLimits | null;
