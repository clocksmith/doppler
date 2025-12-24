/**
 * DOPPLER - Distributed Object Parallel Processing Layer Executing REPLOID
 * High-Performance Offline Browser Inference
 * Main entry point for the DOPPLER subsystem.
 */

export const DOPPLER_VERSION = '0.1.0';

// Memory & Capability (Agent-A)
export { getMemoryCapabilities } from './memory/capability.js';
export { detectUnifiedMemory } from './memory/unified-detect.js';
export { HeapManager, getHeapManager } from './memory/heap-manager.js';
export { AddressTable } from './memory/address-table.js';

// Storage & Format (Agent-B)
export {
  parseManifest,
  getManifest,
  getShardInfo,
  getShardCount,
  isMoE,
  getShardsForExpert,
} from './storage/rdrr-format.js';
export {
  initOPFS,
  openModelDirectory,
  loadShard,
  loadShardSync,
  writeShard,
  verifyIntegrity,
  listModels,
  deleteModel,
} from './storage/shard-manager.js';
export { downloadModel } from './storage/downloader.js';
export {
  requestPersistence,
  getQuotaInfo,
  getStorageReport,
  checkSpaceAvailable,
} from './storage/quota.js';

// GPU & Kernels (Agent-C)
export {
  initDevice,
  getDevice,
  getKernelCapabilities,
  getDeviceLimits,
  destroyDevice,
} from './gpu/device.js';
export {
  runMatmul,
  dequantize,
  runAttention,
  runRMSNorm,
  runSoftmax,
  runRoPE,
  runSiLU,
  runGather,
  runResidualAdd,
  selectMatmulKernel,
  selectDequantKernel,
  getTunedWorkgroupSize,
  autoTuneKernels,
} from './gpu/kernel-selector.js';
export {
  getBufferPool,
  createStagingBuffer,
  acquireBuffer,
  releaseBuffer,
  readBuffer,
} from './gpu/buffer-pool.js';
export { GPUProfiler, getProfiler, createProfiler } from './gpu/profiler.js';
export { KernelTuner, getKernelTuner } from './gpu/kernel-tuner.js';

// Inference Pipeline (Agent-D)
export { MoERouter } from './inference/moe-router.js';
export { SpeculativeDecoder } from './inference/speculative.js';
export { InferencePipeline, createPipeline } from './inference/pipeline.js';
export { KVCache } from './inference/kv-cache.js';
export { Tokenizer } from './inference/tokenizer.js';

// Loader
export {
  DopplerLoader,
  getDopplerLoader,
  createDopplerLoader,
} from './loader/doppler-loader.js';

// Native Bridge (Phase 3)
export {
  MAGIC as BRIDGE_MAGIC,
  HEADER_SIZE as BRIDGE_HEADER_SIZE,
  CMD as BRIDGE_CMD,
  FLAGS as BRIDGE_FLAGS,
  encodeMessage,
  decodeHeader,
  createReadRequest,
  createAck,
  ExtensionBridgeClient,
  BridgeStatus,
  getBridgeClient,
  isBridgeAvailable,
  createBridgeClient,
  readFileNative,
} from './bridge/index.js';

// Types re-exports
export type { RDRRManifest, ShardInfo, TensorLocation } from './storage/rdrr-format.js';
export type { MemoryCapabilities } from './memory/capability.js';
export type { GpuCapabilities, GpuLimits } from './gpu/device.js';
export type { SamplingOptions } from './inference/pipeline/sampling.js';
export type { ParsedModelConfig } from './inference/pipeline/config.js';
