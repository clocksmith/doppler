import type { LayerContext } from '../types.js';
export declare function processHyperConnectionLayer(layerIdx: number, input: GPUBuffer, numTokens: number, isPrefill: boolean, context: LayerContext): Promise<GPUBuffer>;
