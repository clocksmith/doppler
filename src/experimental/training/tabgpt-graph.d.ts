import type { TabGPTConfig } from './tabgpt-model.js';

import type { TabGPTOps, TabGPTDispatch } from './tabgpt-ops.js';
import type { TabGPTMatmul } from './tabgpt-matmul.js';

export declare function buildTabGPTGraph(
  config: TabGPTConfig,
  parameters: ReadonlyMap<string, { value: GPUBufferBinding; gradient: GPUBufferBinding }>,
  allocate: (elements: number, label: string) => GPUBuffer,
  op: TabGPTOps['compile'],
  matmul: TabGPTMatmul,
): {
  forward: TabGPTDispatch[];
  backward: TabGPTDispatch[];
  ids: GPUBuffer;
  targets: GPUBuffer;
  outputLoss: GPUBuffer;
};
