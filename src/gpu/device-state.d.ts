export interface SharedDeviceState {
  gpuDevice: GPUDevice | null;
  kernelCapabilities: unknown;
  resolvedPlatformConfig: unknown;
  lastDeviceLossInfo: unknown;
  platformInitialized: boolean;
  deviceEpoch: number;
  bufferOwners: WeakMap<object, GPUDevice>;
  deviceInitPromise: Promise<unknown> | null;
}

export declare function getSharedDeviceState(): SharedDeviceState;
export declare function getSharedDeviceEpoch(): number;
