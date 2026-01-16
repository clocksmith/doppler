import { releaseBuffer } from '../memory/buffer-pool.js';
import { isWeightBuffer, isCpuWeightBuffer } from '../gpu/weight-buffer.js';
import { trace as debugTrace } from '../debug/index.js';

export class LoaderState {
  embeddings = null;
  lmHead = null;
  finalNorm = null;
  layers = new Map();
  experts = new Map();
  gpuBuffers = new Set();
  isLoaded = false;

  setLayer(layerIndex, weights) {
    this.layers.set(layerIndex, weights);
  }

  getLayer(layerIndex) {
    return this.layers.get(layerIndex);
  }

  hasLayer(layerIndex) {
    return this.layers.has(layerIndex);
  }

  getLayerIndices() {
    return Array.from(this.layers.keys()).sort((a, b) => a - b);
  }

  static expertKey(layerIndex, expertIndex) {
    return `${layerIndex}_${expertIndex}`;
  }

  setExpert(layerIndex, expertIndex, weights) {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    this.experts.set(key, weights);
  }

  getExpert(layerIndex, expertIndex) {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    return this.experts.get(key);
  }

  hasExpert(layerIndex, expertIndex) {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    return this.experts.has(key);
  }

  trackBuffer(buffer) {
    this.gpuBuffers.add(buffer);
  }

  trackBuffers(buffers) {
    for (const buffer of buffers) {
      this.gpuBuffers.add(buffer);
    }
  }

  releaseBuffer(buffer) {
    if (this.gpuBuffers.has(buffer)) {
      releaseBuffer(buffer);
      this.gpuBuffers.delete(buffer);
    }
  }

  releaseAllBuffers() {
    for (const buffer of this.gpuBuffers) {
      releaseBuffer(buffer);
    }
    this.gpuBuffers.clear();
  }

  getSnapshot() {
    return {
      isLoaded: this.isLoaded,
      layerCount: this.layers.size,
      expertCount: this.experts.size,
      gpuBufferCount: this.gpuBuffers.size,
      hasEmbeddings: this.embeddings !== null,
      hasLmHead: this.lmHead !== null,
      hasFinalNorm: this.finalNorm !== null,
    };
  }

  hasAnyWeights() {
    return (
      this.embeddings !== null ||
      this.lmHead !== null ||
      this.finalNorm !== null ||
      this.layers.size > 0 ||
      this.experts.size > 0
    );
  }

  clear() {
    debugTrace.loader('Clearing loader state...');

    this.releaseAllBuffers();

    this.embeddings = null;
    this.lmHead = null;
    this.finalNorm = null;

    this.layers.clear();
    this.experts.clear();

    this.isLoaded = false;

    debugTrace.loader('Loader state cleared');
  }

  prepareForLoad() {
    if (this.hasAnyWeights()) {
      debugTrace.loader('Clearing existing state before new load');
      this.clear();
    }
  }

  markLoaded() {
    this.isLoaded = true;
  }

  static getGPUBuffer(weight) {
    if (!weight) return null;
    if (weight instanceof GPUBuffer) return weight;
    if (isWeightBuffer(weight)) return weight.buffer;
    return null;
  }

  static isGPUBacked(weight) {
    if (!weight) return false;
    if (weight instanceof GPUBuffer) return true;
    if (isWeightBuffer(weight)) return true;
    if (isCpuWeightBuffer(weight)) return false;
    if (weight instanceof Float32Array) return false;
    return false;
  }
}

export function createLoaderState() {
  return new LoaderState();
}
