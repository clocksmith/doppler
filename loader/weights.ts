/**
 * Shared weight shape definitions for loader modules.
 *
 * @module loader/weights
 */

export interface ExpertWeights {
  gate?: GPUBuffer | Float32Array | null;
  up?: GPUBuffer | Float32Array | null;
  down?: GPUBuffer | Float32Array | null;
  isGptOss?: boolean;
  expertIdx?: number;
  numExperts?: number;
  gateUpBlocks?: GPUBuffer | null;
  gateUpScales?: GPUBuffer | null;
  gateUpBias?: GPUBuffer | null;
  downBlocks?: GPUBuffer | null;
  downScales?: GPUBuffer | null;
  downBias?: GPUBuffer | null;
}
