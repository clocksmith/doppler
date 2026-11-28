import type { TabGPTBuffer, TabGPTDispatch } from './tabgpt-ops.js';

export type TabGPTMatmul = (
  a: TabGPTBuffer, b: TabGPTBuffer, out: TabGPTBuffer,
  m: number, n: number, k: number, transpose?: '' | 'a' | 'b',
) => TabGPTDispatch;

export function createTabGPTMatmul(
  device: GPUDevice,
  allocateBuffer: (size: number, usage: GPUBufferUsageFlags, label: string) => GPUBuffer,
): Promise<TabGPTMatmul>;
