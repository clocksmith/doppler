export { detectSandwichNorm, hasConvLayers } from "./layer.js";
export function assertSupportedLayerRuntime(layerIdx: number, config: import('../config.js').ParsedModelConfig | null): void;
export function isLinearLayerType(layerType: unknown): boolean;
export function shouldUseSandwichRMSNormPairFusion(options: {
  context: import('../types.js').LayerContext;
  sandwichNorm: import('../types.js').SandwichNormInfo;
  layerWeights: import('../types.js').LayerWeights;
  numTokens: number;
  hiddenSize: number;
  attnOutput: import('../../../../gpu/tensor.js').Tensor;
  inputTensor: import('../../../../gpu/tensor.js').Tensor | null;
}): boolean;
