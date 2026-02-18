import type { MoEConfig, MoEExpertWeights } from './moe-impl.d.ts';

export declare function runGptOssExpertCPU(
  layerIdx: number,
  expertIdx: number,
  input: Float32Array,
  config: MoEConfig,
  expertWeights: Map<string, MoEExpertWeights>
): Promise<Float32Array>;
