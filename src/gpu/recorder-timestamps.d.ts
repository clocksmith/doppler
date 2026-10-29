/** Allocates the timestamp query and readback resources atomically. Partial
 * allocation is destroyed on failure; a successful caller owns all resources.
 */
export function createRecorderTimestampResources(device: GPUDevice & { limits: GPUSupportedLimits & { maxQuerySetSize?: number } }, label: string): {
  querySet: GPUQuerySet; queryBuffer: GPUBuffer; readbackBuffer: GPUBuffer; capacity: number;
};
