import type { MoERouter } from '../../moe-router.js';
import type { MoEConfig, MoEExpertWeights, ExpertLoader, LayerRouterWeights } from './moe-impl.d.ts';

export interface ActiveExpertSchedule {
  selection: 'all' | 'topk-readback';
  activeExperts: number[];
  tokenCounts: Uint32Array | null;
}

export declare function buildActiveExpertScheduleFromIndices(
  indices: Uint32Array,
  numExperts: number,
  maxTokensPerExpert: number,
  selection?: 'topk-readback'
): ActiveExpertSchedule;

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
