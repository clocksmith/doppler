import type { ResourceScope } from '../../../resource-scope.js';
export interface DecodeStepRecorderCleanup {
  abort(): unknown | Promise<unknown>;
  completeDeferredCleanup(options: { discardPooled: boolean }): unknown | Promise<unknown>;
}
export function createDecodeStepResources(
  recorder: DecodeStepRecorderCleanup | undefined,
  release: (buffer: GPUBuffer) => void,
  discard: (buffer: GPUBuffer) => void
): { resources: ResourceScope<GPUBuffer>; close(failed: boolean, failure: unknown): Promise<void> };
