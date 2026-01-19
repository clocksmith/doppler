/**
 * capabilities-detector.d.ts - Browser/GPU capability detection
 *
 * @module app/capabilities-detector
 */

export interface CapabilitiesState {
  webgpu: boolean;
  f16: boolean;
  subgroups: boolean;
  memory64: boolean;
}

export interface AdapterInfoLike {
  vendor?: string | null;
  device?: string | null;
  architecture?: string | null;
  description?: string | null;
}

export interface GPULimits {
  maxBufferSize: number;
  maxStorageSize: number;
}

export declare class CapabilitiesDetector {
  detect(): Promise<CapabilitiesState>;
  getState(): CapabilitiesState;
  getAdapter(): GPUAdapter | null;
  getAdapterInfo(): GPUAdapterInfo | AdapterInfoLike | null;
  resolveGPUName(info: AdapterInfoLike): string;
  isUnifiedMemoryArchitecture(info: AdapterInfoLike): boolean;
  getGPULimits(): GPULimits | null;
  hasTimestampQuery(): boolean;
}
