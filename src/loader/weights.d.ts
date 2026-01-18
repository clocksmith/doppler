/**
 * Shared weight shape definitions for loader modules.
 *
 * @module loader/weights
 */

import type { WeightBuffer } from '../gpu/weight-buffer.js';

export interface ExpertWeights {
  gate?: GPUBuffer | WeightBuffer | Float32Array | null;
  up?: GPUBuffer | WeightBuffer | Float32Array | null;
  down?: GPUBuffer | WeightBuffer | Float32Array | null;
  expertFormat?: 'mixtral' | 'gpt-oss';
  expertIdx?: number;
  numExperts?: number;
  gateUpBlocks?: GPUBuffer | null;
  gateUpScales?: GPUBuffer | null;
  gateUpBias?: GPUBuffer | null;
  downBlocks?: GPUBuffer | null;
  downScales?: GPUBuffer | null;
  downBias?: GPUBuffer | null;
}
