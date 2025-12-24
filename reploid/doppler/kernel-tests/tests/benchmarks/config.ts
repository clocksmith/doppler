/**
 * Benchmark Configuration
 *
 * Standard workload sizes based on common LLM configurations
 */

/**
 * Model configuration
 */
export interface ModelConfig {
  name: string;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  numExperts: number;
  topK: number;
  vocabSize: number;
}

/**
 * Sequence length configuration
 */
export interface SeqConfig {
  seqLen: number;
  name: string;
}

/**
 * Batch size configuration
 */
export interface BatchConfig {
  batchSize: number;
  name: string;
}

/**
 * Workload configuration
 */
export interface WorkloadConfig extends ModelConfig, SeqConfig, BatchConfig {
  name: string;
  model: string;
  seq: string;
  batch: string;
  M?: number;
  K?: number;
  N?: number;
  numTokens?: number;
  kvLen?: number;
}

/**
 * Benchmark settings
 */
export interface BenchmarkSettings {
  warmupIterations: number;
  timedIterations: number;
  minTime: number;
  maxTime: number;
}

/**
 * Benchmark matrix entry
 */
export interface BenchmarkMatrixEntry {
  model: string;
  seq: string;
  batch?: string;
}

/**
 * Model configurations (Llama-style)
 */
export const MODEL_CONFIGS: Record<string, ModelConfig> = {
  // Small model (debugging)
  small: {
    name: 'Small',
    hiddenSize: 512,
    intermediateSize: 1408, // 2.75x
    numHeads: 8,
    numKVHeads: 8,
    headDim: 64,
    numExperts: 4,
    topK: 2,
    vocabSize: 32000,
  },

  // 7B-class model
  '7b': {
    name: '7B',
    hiddenSize: 4096,
    intermediateSize: 11008,
    numHeads: 32,
    numKVHeads: 32, // MHA
    headDim: 128,
    numExperts: 8,
    topK: 2,
    vocabSize: 32000,
  },

  // Mixtral 8x7B style
  'mixtral': {
    name: 'Mixtral-8x7B',
    hiddenSize: 4096,
    intermediateSize: 14336,
    numHeads: 32,
    numKVHeads: 8, // GQA
    headDim: 128,
    numExperts: 8,
    topK: 2,
    vocabSize: 32000,
  },

  // Large model (13B-class)
  '13b': {
    name: '13B',
    hiddenSize: 5120,
    intermediateSize: 13824,
    numHeads: 40,
    numKVHeads: 40,
    headDim: 128,
    numExperts: 8,
    topK: 2,
    vocabSize: 32000,
  },
};

/**
 * Sequence length configurations
 */
export const SEQ_CONFIGS: Record<string, SeqConfig> = {
  single: { seqLen: 1, name: 'Single token (decode)' },
  short: { seqLen: 32, name: 'Short prompt' },
  medium: { seqLen: 128, name: 'Medium prompt' },
  long: { seqLen: 512, name: 'Long prompt' },
  veryLong: { seqLen: 2048, name: 'Very long prompt' },
};

/**
 * Batch size configurations
 */
export const BATCH_CONFIGS: Record<string, BatchConfig> = {
  single: { batchSize: 1, name: 'Single' },
  small: { batchSize: 4, name: 'Small batch' },
  medium: { batchSize: 16, name: 'Medium batch' },
  large: { batchSize: 64, name: 'Large batch' },
};

/**
 * Benchmark settings
 */
export const BENCHMARK_SETTINGS: BenchmarkSettings = {
  // Number of warmup iterations
  warmupIterations: 5,

  // Number of timed iterations
  timedIterations: 20,

  // Minimum time to run (ms)
  minTime: 1000,

  // Maximum time per benchmark (ms)
  maxTime: 30000,
};

/**
 * Get workload configuration for a specific kernel
 */
export function getWorkloadConfig(kernel: string, model: string, seq: string, batch: string = 'single'): WorkloadConfig {
  const m = MODEL_CONFIGS[model];
  const s = SEQ_CONFIGS[seq];
  const b = BATCH_CONFIGS[batch];

  const config: WorkloadConfig = {
    name: `${kernel} - ${m.name} - ${s.name} - ${b.name}`,
    model,
    seq,
    batch,
    ...m,
    ...s,
    ...b,
  };

  // Compute derived values based on kernel
  switch (kernel) {
    case 'matmul':
      // For linear projections: [batch * seq, hidden] @ [hidden, out]
      config.M = b.batchSize * s.seqLen;
      config.K = m.hiddenSize;
      config.N = m.hiddenSize;
      break;

    case 'attention':
      // Attention: [batch * heads, seq, headDim] ops
      config.numTokens = b.batchSize * s.seqLen;
      config.kvLen = s.seqLen; // For prefill
      break;

    case 'ffn':
      // FFN: [batch * seq, hidden] -> [batch * seq, intermediate]
      config.M = b.batchSize * s.seqLen;
      config.K = m.hiddenSize;
      config.N = m.intermediateSize;
      break;

    case 'topk':
    case 'scatter_add':
    case 'moe':
      // MoE: [batch * seq] tokens routing to experts
      config.numTokens = b.batchSize * s.seqLen;
      break;

    case 'softmax':
    case 'rmsnorm':
      config.numTokens = b.batchSize * s.seqLen;
      break;
  }

  return config;
}

/**
 * Standard benchmark matrix
 */
export const BENCHMARK_MATRIX: Record<string, BenchmarkMatrixEntry[]> = {
  matmul: [
    { model: 'small', seq: 'single' },
    { model: 'small', seq: 'medium' },
    { model: '7b', seq: 'single' },
    { model: '7b', seq: 'short' },
    { model: '7b', seq: 'medium' },
    { model: 'mixtral', seq: 'single' },
    { model: 'mixtral', seq: 'medium' },
  ],

  attention: [
    { model: 'small', seq: 'short' },
    { model: 'small', seq: 'medium' },
    { model: '7b', seq: 'short' },
    { model: '7b', seq: 'medium' },
    { model: 'mixtral', seq: 'short' }, // GQA
  ],

  topk: [
    { model: 'small', seq: 'medium' },
    { model: 'mixtral', seq: 'single' },
    { model: 'mixtral', seq: 'short' },
    { model: 'mixtral', seq: 'medium' },
  ],

  moe: [
    { model: 'mixtral', seq: 'single' },
    { model: 'mixtral', seq: 'short' },
    { model: 'mixtral', seq: 'medium' },
  ],
};

export default {
  MODEL_CONFIGS,
  SEQ_CONFIGS,
  BATCH_CONFIGS,
  BENCHMARK_SETTINGS,
  getWorkloadConfig,
  BENCHMARK_MATRIX,
};
