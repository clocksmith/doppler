/** @deprecated Since 0.7.0. Removed in 0.8.0; no replacement export. Node LoRA loading: `doppler-gpu/training`. */
export * from '../../inference/tokenizer.js';
export * from '../../inference/kv-cache.js';
export { SpeculativeDecoder } from '../../inference/speculative.js';
export type {
  DraftModel,
  MainModel,
  SpeculativeConfig,
  VerificationResult,
  StepResult,
  DecodingStats,
  StatsWithSpeedup,
} from '../../inference/speculative.js';
export * from '../../inference/expert-router.js';
export * from '../../inference/moe-router.js';
export * from '../../inference/multi-model-network.js';
export * from '../../inference/multi-pipeline-pool.js';
export * from '../../gpu/kernels/logit-merge.js';
export * from '../../inference/pipelines/structured/json-head-pipeline.js';
export * from './energy-row-head-pipeline.js';
export { buildConservativeMultimodalGenerationOptions } from '../../inference/pipelines/text.js';
export * from '../adapters/index.js';
