export interface DeviceResidency {
  readonly bufferOwners: WeakMap<object, GPUDevice>;
  readonly lostDevices: WeakSet<GPUDevice>;
}

export declare function markDeviceLost(device: GPUDevice | null | undefined): void;
export declare function observeDeviceLoss(device: GPUDevice | null | undefined): void;
export declare function registerBufferDevice(buffer: GPUBuffer, device: GPUDevice): void;
export declare function getBufferDevice(buffer: GPUBuffer): GPUDevice | null;
export declare function isDeviceLost(device: GPUDevice | null | undefined): boolean;
export declare function recordCreatedBuffer(buffer: GPUBuffer, device: GPUDevice): void;
