import type { WeightBuffer, CpuWeightBuffer } from '../gpu/weight-buffer.js';
import type { LayerWeights } from './loader-types.js';
import type { ExpertWeights } from './weights.js';

export type WeightType = GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;

export interface LoaderStateSnapshot {
  isLoaded: boolean;
  layerCount: number;
  expertCount: number;
  gpuBufferCount: number;
  hasEmbeddings: boolean;
  hasLmHead: boolean;
  hasFinalNorm: boolean;
}

export class LoaderState {
  embeddings: WeightType;
  lmHead: WeightType;
  finalNorm: GPUBuffer | Float32Array | null;
  layers: Map<number, LayerWeights>;
  experts: Map<string, ExpertWeights>;
  gpuBuffers: Set<GPUBuffer>;
  isLoaded: boolean;

  setLayer(layerIndex: number, weights: LayerWeights): void;
  getLayer(layerIndex: number): LayerWeights | undefined;
  hasLayer(layerIndex: number): boolean;
  getLayerIndices(): number[];

  static expertKey(layerIndex: number, expertIndex: number): string;
  setExpert(layerIndex: number, expertIndex: number, weights: ExpertWeights): void;
  getExpert(layerIndex: number, expertIndex: number): ExpertWeights | undefined;
  hasExpert(layerIndex: number, expertIndex: number): boolean;

  trackBuffer(buffer: GPUBuffer): void;
  trackBuffers(buffers: GPUBuffer[]): void;
  releaseBuffer(buffer: GPUBuffer): void;
  releaseAllBuffers(): void;

  getSnapshot(): LoaderStateSnapshot;
  hasAnyWeights(): boolean;
  clear(): void;
  prepareForLoad(): void;
  markLoaded(): void;

  static getGPUBuffer(weight: WeightType): GPUBuffer | null;
  static isGPUBacked(weight: WeightType): boolean;
}

export function createLoaderState(): LoaderState;
