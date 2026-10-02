export interface DeviceMemorySnapshot {
  scope: 'device-gpu-buffers'; maxBytes: number | null; liveBytes: number; peakBytes: number;
  allocations: number; rejected: number;
  categories: Record<'weights' | 'attention' | 'recurrent' | 'other', number>;
  labels: Array<{label: string; category: string; bytes: number; count: number}>;
}
export function installDeviceMemoryAccounting(device: GPUDevice): void;
export function setDeviceMemoryBudget(device: GPUDevice, maxBytes: number | null): DeviceMemorySnapshot;
export function markDeviceWeights(device: GPUDevice, buffers: Iterable<GPUBuffer>): void;
export function getDeviceMemorySnapshot(device: GPUDevice | null): DeviceMemorySnapshot | null;
