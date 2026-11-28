import type { TabGPTConfig } from './tabgpt-model.js';

export type TabGPTBuffer = GPUBuffer | GPUBufferBinding;
export type TabGPTDispatch = (pass: GPUComputePassEncoder) => void;

export interface TabGPTOpDimensions {
  width?: number;
  count?: number;
  offset?: number;
}

export interface TabGPTOps {
  compile(
    name: string,
    buffers: TabGPTBuffer[],
    dims?: TabGPTOpDimensions,
  ): TabGPTDispatch;
}

export function createTabGPTOps(device: GPUDevice, config: TabGPTConfig): Promise<TabGPTOps>;
