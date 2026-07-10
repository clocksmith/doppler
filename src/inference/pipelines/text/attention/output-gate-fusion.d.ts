import type { Tensor } from '../../../../gpu/tensor.js';
import type { KernelPathSchema } from '../../../../config/schema/index.js';

export interface AttentionOutputGateFusionOptions {
  session?: {
    attentionDecodeOnline?: {
      useOutputGateFusion?: boolean;
    } | null;
  } | null;
  qGateTensor?: Tensor | null;
  numTokens: number;
  numHeads: number;
  headDim: number;
  cachedKDtype: string;
  cachedVDtype: string;
  kernelPath?: KernelPathSchema | null;
  diffusionGemmaDecoder?: boolean;
}

export function canUseAttentionOutputGateFusion(options?: AttentionOutputGateFusionOptions): boolean;
