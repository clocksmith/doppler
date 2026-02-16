import type { MoERouter } from '../../moe-router.js';
import type { MoEConfig, MoEExpertWeights, ExpertLoader, LayerRouterWeights } from './moe-impl.d.ts';

export declare function moeFeedForwardGPU(
  inputBuffer: GPUBuffer,
  numTokens: number,
  config: MoEConfig,
  moeRouter: MoERouter,
  expertWeights: Map<string, MoEExpertWeights>,
  expertLoader: ExpertLoader,
  layerIdx: number,
  layerRouterWeights?: Map<number, LayerRouterWeights>
): Promise<GPUBuffer>;
