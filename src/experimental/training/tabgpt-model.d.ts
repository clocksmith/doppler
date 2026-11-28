/** Geometry and optimizer schedule for the FP32, tied-weight TabGPT architecture. */
export interface TabGPTConfig {
  B: number;
  T: number;
  C: number;
  L: number;
  NH: number;
  V: number;
  lr: number;
  warmup: number;
  f16: false;
}

export interface TabGPTTrainer {
  readonly step: number;
  readonly numParams: number;
  readonly parameterNames: readonly string[];
  readonly memory: { ownedBufferBytes: number; peakOwnedBufferBytes: number };
  trainStep(inputs: Uint32Array, targets: Uint32Array): Promise<number>;
  evaluate(inputs: Uint32Array, targets: Uint32Array): Promise<number>;
  readParameter(name: string): Promise<Float32Array>;
  destroy(): void;
}

export function createTabGPTTrainer(
  device: GPUDevice,
  configuration: TabGPTConfig,
  initialWeights?: Map<string, Float32Array>,
): Promise<TabGPTTrainer>;
