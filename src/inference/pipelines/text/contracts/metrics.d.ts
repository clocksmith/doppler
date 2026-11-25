import type { ParsedModelConfig } from '../config.js';
import type { LoRAAdapter } from '../lora-types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../../gpu/weight-buffer.js';
import type { ProbeConfigSchema, KernelPathSchema } from '../../../../config/schema/index.js';
import type { ExpertLoader } from '../moe-impl.js';
import type { MoERouter } from '../../../moe-router.js';
import type { DecodeBufferManager } from '../../../decode-buffers.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { CompiledLayerPipeline } from "../execution/layer-plan.js";
import type { WeightBufferConfig, WeightDebugFlags, LayerWeights, ExpertWeights, RouterWeights } from "../loading/weights.js";
import type {
  ExecutionV1PerLayerInputsSessionSchema,
  ExecutionV1PoliciesSchema,
} from "../../../../config/schema/execution/execution-v1.schema.js";
import type { KVCache } from '../../../kv-cache/base.js';
import type { SlidingWindowKVCache } from '../../../kv-cache/sliding-window.js';
import type { TieredKVCache } from '../../../kv-cache/tiered.js';
import type { BasisDecomposedPagedCache } from '../../../kv-cache/basis-decomposed-paged.js';
import type { MixedGeometryKVCache } from '../../../kv-cache/mixed-geometry.js';
import type { DecodeRingStats } from '../../../decode-ring.js';
import type { LinearAttentionRuntime } from "../linear-attention/linear-attention.js";
import type { TokenizerLoadTiming } from '../../../tokenizer.js';
import type { LoaderLoadTiming, PerLayerInputWeights } from '../../../../loader/loader-types.js';
import type { UniformCacheStats } from '../../../../gpu/uniform-cache.js';
import type { Tensor } from '../../../../gpu/tensor.js';

export interface WorkloadPhaseTiming {
  [key: string]: number | string | boolean | null | Record<string, number> | WorkloadPhaseTiming | WorkloadPhaseTiming[];
}

// ============================================================================
// Statistics Types
// ============================================================================

/**
 * Pipeline performance statistics.
 */
export type PipelineLoadTimingPhase =
  | 'reset'
  | 'configResolution'
  | 'kernelWarmup'
  | 'tokenizer'
  | 'executionSetup'
  | 'loadWeights'
  | 'rope'
  | 'convStates'
  | 'warmGeneration';

export interface PipelineLoadTiming {
  schemaVersion: 1;
  source: 'doppler-pipeline';
  modelId: string | null;
  status: 'running' | 'complete' | 'failed';
  phasesMs: Record<PipelineLoadTimingPhase, number | null>;
  details: {
    tokenizer: TokenizerLoadTiming | null;
    warmGeneration: import("../generator/session/warm-generation.js").WarmGenerationResult | null;
  };
  totalMs: number | null;
}

export interface PipelineStats {
  reusedPrefixTokens?: number;
  prefillComputedTokens?: number;
  /** Per-generation whole-token counters; dispatches is null until a program is built. */
  wholeTokenProgram?: import("../generator/whole-token/whole-token-program.js").WholeTokenProgram['stats'] & {
    dispatches: number | null;
  };
  /** Per-generation wall time of replayed whole-token decode steps, in milliseconds. */
  wholeTokenProgramWallMs?: number;
  /** Full raw decoding counts when a manifest declares final-answer output semantics. */
  outputUsage?: { rawCharacters: number; rawTokens: number; finalCharacters: number; state: string };
  /** Total prefill time in milliseconds */
  prefillTimeMs: number;

  /** Time to first token in milliseconds */
  ttftMs: number;

  /** Total decode time in milliseconds */
  decodeTimeMs: number;

  /** Number of tokens processed in prefill */
  prefillTokens: number;

  /** Number of tokens generated in decode */
  decodeTokens: number;

  /** Memory usage in bytes */
  memoryUsageBytes: number;

  // Fields from pipeline.ts
  modelLoadMs?: number;
  loadTiming?: LoaderLoadTiming | null;
  pipelineLoadTiming?: PipelineLoadTiming | null;
  speculationAttempts?: number;
  speculationAccepted?: number;
  tokensGenerated: number;
  totalTimeMs: number;
  gpuTimePrefillMs?: number;
  gpuTimeDecodeMs?: number;
  decodeRecordMs?: number;
  decodeRecordOps?: number;
  decodeRecordPasses?: number;
  decodeRecordOpLabels?: Record<string, number>;
  prefillRecordMs?: number;
  prefillRecordOps?: number;
  prefillRecordPasses?: number;
  prefillRecordOpLabels?: Record<string, number>;
  uniformCache?: UniformCacheStats | null;
  decodeSubmitWaitMs?: number;
  decodeReadbackWaitMs?: number;
  decodeReadbackMapWaitMs?: number;
  decodeReadbackCleanupMs?: number;
  decodeReadbackCopyMs?: number;
  decodeMode?: 'single-token' | 'batched-gpu' | 'batched-gpu-stepwise-ple' | null;
  batchGuardReason?: string | null;
  singleTokenSubmitWaitMs?: number;
  singleTokenReadbackWaitMs?: number;
  singleTokenReadbackMapWaitMs?: number;
  singleTokenReadbackCleanupMs?: number;
  singleTokenReadbackCopyMs?: number;
  singleTokenOrchestrationMs?: number;
  batching?: BatchingStats | null;
  plePreparedTokenCacheHits?: number;
  plePreparedTokenCacheMisses?: number;
  plePreparedTokenCacheEntries?: number;
  plePreparedTokenCacheBytes?: number;
  pleHotVocabularyHits?: number;
  pleHotVocabularyMisses?: number;
  decodeRing?: DecodeRingStats | null;
  executionPlan?: {
    primary: {
      id: string;
      kernelPathId: string | null;
      kernelPathSource: string;
      activationDtype: string;
      readbackInterval: number | null;
      readbackMode: string | null;
      maxBatchDecodeTokens?: number | null;
      batchSize: number;
      stopCheckMode: string;
      disableCommandBatching?: boolean;
      ringTokens: number | null;
      ringStop: number | null;
      ringStaging: number | null;
    } | null;
    fallback: {
      id: string;
      kernelPathId: string | null;
      kernelPathSource: string;
      activationDtype: string;
      readbackInterval: number | null;
      readbackMode: string | null;
      maxBatchDecodeTokens?: number | null;
      batchSize: number;
      stopCheckMode: string;
      disableCommandBatching?: boolean;
      ringTokens: number | null;
      ringStop: number | null;
      ringStaging: number | null;
    } | null;
    activePlanIdAtStart: string | null;
    finalActivePlanId: string | null;
    transitions: Array<{
      kind: string;
      reason: string | null;
      decodeStep: number;
      seqLen: number;
      fromPlanId: string | null;
      toPlanId: string | null;
      fromKernelPathId: string | null;
      toKernelPathId: string | null;
    }>;
  } | null;
  kernelPathId?: string | null;
  kernelPathSource?: string | null;
  /** Immutable semantic attention plans captured by observation policy */
  attentionPlans?: Array<import('../attention/plan.js').SemanticAttentionPlan>;
  /** Tagged attention ownership/lifetime events */
  attentionResourceEvents?: Array<import('../../../resource-scope.js').ResourceEvent>;
  /** Canonical behavior receipts captured by observation policy */
  refactorReceipts?: Array<import('../../../refactor-receipt.js').RefactorReceipt>;
  prefillProfileSteps?: Array<{
    label?: string;
    timings: Record<string, number>;
    totalMs?: number;
  }>;
  decodeProfileSteps?: Array<{
    step?: number;
    stepStart?: number;
    stepCount?: number;
    batch?: boolean;
    timings: Record<string, number>;
    totalMs?: number;
  }>;
}

/**
 * Batching-specific statistics.
 */
export interface BatchingStats {
  /** Number of batched forward passes */
  batchedForwardCalls: number;

  /** Number of unbatched forward passes */
  unbatchedForwardCalls: number;

  /** Total time in batched mode */
  totalBatchedTimeMs: number;

  /** Total time in unbatched mode */
  totalUnbatchedTimeMs: number;

  /** Number of GPU command submissions */
  gpuSubmissions: number;

  /** Largest requested decode burst in tokens before runtime safety bounding */
  requestedBatchTokens?: number;

  /** Largest actual decode burst submitted in tokens after runtime safety bounding */
  effectiveBatchTokens?: number;

  /** Total GPU batch-path decode tokens recorded/submitted for execution */
  executedBatchTokens?: number;

  /** Total batch-path decode tokens retained by stop resolution */
  resolvedBatchTokens?: number;

  /** Active max decode-burst cap selected by execution rules, when present */
  maxBatchTokenCap?: number | null;

  /** Number of decode loop iterations where requested burst tokens were clamped */
  batchClampCount?: number;
}
