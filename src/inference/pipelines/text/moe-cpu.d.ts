import type { MoERouter } from '../../moe-router.js';
import type { ExpertWeights } from './types.js';
import type { MoEConfig, MoEExpertWeights, ExpertLoader } from './moe-impl.d.ts';

export declare function moeFeedForwardCPU(
  hiddenStates: Float32Array,
  numTokens: number,
  config: MoEConfig,
  moeRouter: MoERouter,
  expertWeights: Map<string, MoEExpertWeights>,
  expertLoader: ExpertLoader,
  layerIdx: number
): Promise<Float32Array>;
