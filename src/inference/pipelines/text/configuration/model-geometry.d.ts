export { resolveLayerIntermediateSize } from '../config.js';
export function resolveRotaryDim(headDim: number, partialRotaryFactor: unknown, modelId: string): number;
export function resolveFrequencyBaseDim(headDim: number, rotaryDim: number, frequencyBaseDim: unknown, modelId: string, fieldName: string): number;
export function buildPerLayerIntermediateSizes(config: {
  baseIntermediateSize: number; numLayers: number; numKvSharedLayers: number | null;
  useDoubleWideMlp: boolean; modelId: string;
}): number[];
