import type { InferencePipeline, KVCacheSnapshot, PromptInput } from '../../inference/pipelines/text.js';
import type { ChatMessage } from '../../inference/pipelines/text/chat-format.js';
import type { GenerateOptions } from '../../generation/index.js';
import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { LogitsStepResult, PipelineStats, PrefillResult, SequenceEncodeOptions, SequenceEncodeResult } from '../../inference/pipelines/text/types.d.ts';
import type { LoRAManifest } from './types.js';
import type { LoRALoadOptions } from './lora.js';
import type { DopplerPersistentCacheReceipt } from './model-source.js';
import type { DopplerResolutionPolicy, ResolvedDopplerResolutionPolicy } from './resolution-policy.js';
import type {
  DopplerComparisonFingerprint,
  DopplerInspectionTokenRecord,
  DopplerObservationPolicy,
  DopplerWordQualityRecord,
} from '../inspection.js';

export type DopplerGenerateOptions = Omit<GenerateOptions, 'stopTokens'>;

export interface DopplerGenerationConfigEvidence {
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  repetitionPenaltyWindow: number;
  greedyThreshold: number;
  suppressSpecialTokens: boolean;
  suppressSpecialLikeTokens: boolean;
  suppressTokenIds: number[];
  stopSequences: string[];
  useChatTemplate: boolean;
  useSpeculative: boolean | null;
  seed: number | null;
}

export interface DopplerGenerationBackendIdentity {
  backend: 'webgpu';
  adapter: {
    vendor: string | null;
    architecture: string | null;
    device: string | null;
    description: string | null;
  };
  hasF16: boolean;
  hasSubgroups: boolean;
  maxBufferSize: number;
  deviceEpoch: number;
  kernelPathId: string | null;
  kernelPathSource: string | null;
  executionPlanId: string | null;
  activationDtype: string | null;
}

export interface DopplerResolutionIdentity {
  schema: 'doppler.resolution-identity/v1';
  logicalModelId: string;
  resolvedArtifactVariantId: `sha256:${string}`;
  resolvedExecutionId: `sha256:${string}`;
}

export interface DopplerResolvedExecutionIdentity {
  schema: 'doppler.resolved-execution-identity/v1';
  runtime: {
    package: 'doppler-gpu';
    version: string;
    surface: 'node' | 'browser';
  };
  resolvedRuntimeSessionId: `sha256:${string}`;
  activeAdapter: string | null;
  activeAdapterId: string | null;
  activeAdapterDigest: `sha256:${string}` | null;
  backendIdentity: DopplerGenerationBackendIdentity;
}

export interface DopplerGenerationEvidence {
  schema: 'doppler_generation_evidence/v1';
  outputText: string;
  tokenIds: number[];
  transcript: {
    schema: 'doppler_generation_transcript/v1';
    outputText: string;
    tokenIds: number[];
  };
  transcriptHash: string;
  generationConfig: DopplerGenerationConfigEvidence;
  generationConfigHash: string;
  resolution: DopplerResolutionIdentity;
  executionIdentity: DopplerResolvedExecutionIdentity;
  runtimeProfile: {
    schema: 'doppler_runtime_profile/v1';
    runtime: {
      package: 'doppler-gpu';
      version: string;
      surface: 'node' | 'browser';
    };
    model: {
      modelId: string | null;
      manifestHash: string | null;
      activeAdapter: string | null;
      activeAdapterId: string | null;
      activeAdapterDigest: `sha256:${string}` | null;
    };
    resolvedRuntimeSessionId: `sha256:${string}`;
    backendIdentity: DopplerGenerationBackendIdentity;
  };
  runtimeProfileHash: string;
  backendIdentity: DopplerGenerationBackendIdentity;
  backendIdentityHash: string;
  stats: PipelineStats | null;
}

export interface DopplerEmbeddingEvidence {
  schema: 'doppler_embedding_evidence/v1';
  embedding: Float32Array;
  tokens: number[];
  seqLen: number;
  embeddingMode: string;
  phase?: unknown;
  inputHash: `sha256:${string}`;
  outputHash: `sha256:${string}`;
  resolution: DopplerResolutionIdentity;
  executionIdentity: DopplerResolvedExecutionIdentity;
  backendIdentity: DopplerGenerationBackendIdentity;
  backendIdentityHash: `sha256:${string}`;
  stats: PipelineStats | null;
}

export interface DopplerRerankScore {
  index: number;
  document: string;
  score: number;
  probability: number;
  trueLogit: number;
  falseLogit: number;
  tokenCount: number;
  scoringPath: string;
}

export interface DopplerRerankEvidence {
  schema: 'doppler_rerank_evidence/v1';
  query: string;
  documents: string[];
  scores: DopplerRerankScore[];
  ranking: Array<DopplerRerankScore & { rank: number }>;
  inputHash: `sha256:${string}`;
  outputHash: `sha256:${string}`;
  resolution: DopplerResolutionIdentity;
  executionIdentity: DopplerResolvedExecutionIdentity;
  backendIdentity: DopplerGenerationBackendIdentity;
  backendIdentityHash: `sha256:${string}`;
  stats: PipelineStats | null;
}

export interface DopplerChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  evidence: DopplerGenerationEvidence;
}

export interface DopplerModelHandle {
  generate(prompt: string, options?: DopplerGenerateOptions): AsyncGenerator<string, void, void>;
  generateText(prompt: string, options?: DopplerGenerateOptions): Promise<string>;
  generateWithEvidence(
    prompt: string | ChatMessage[],
    options?: DopplerGenerateOptions
  ): Promise<DopplerGenerationEvidence>;
  chat(messages: ChatMessage[], options?: DopplerGenerateOptions): AsyncGenerator<string, void, void>;
  chatText(messages: ChatMessage[], options?: DopplerGenerateOptions): Promise<DopplerChatResponse>;
  embed(prompt: string, options?: Record<string, unknown>): Promise<unknown>;
  embedWithEvidence(
    prompt: string,
    options?: Record<string, unknown>
  ): Promise<DopplerEmbeddingEvidence>;
  embedBatch(prompts: string[], options?: Record<string, unknown>): Promise<unknown>;
  rerankWithEvidence(
    query: string,
    documents: string[],
    options?: { benchmark?: boolean }
  ): Promise<DopplerRerankEvidence>;
  encodeSequence(sequence: string, options?: SequenceEncodeOptions): Promise<SequenceEncodeResult>;
  resetGenerationState(): void;
  loadLoRA(adapter: LoRAManifest | RDRRManifest | string, loadOptions?: LoRALoadOptions): Promise<void>;
  activateLoRAFromTrainingOutput(
    trainingOutput:
      | string
      | {
        adapter?: LoRAManifest | RDRRManifest | string;
        adapterManifest?: LoRAManifest | RDRRManifest;
        adapterManifestJson?: string;
        adapterManifestUrl?: string;
        adapterManifestPath?: string;
      }
      | null
      | undefined
  ): Promise<{
    activated: boolean;
    adapterName: string | null;
    source: string | null;
    reason: string | null;
  }>;
  unloadLoRA(): Promise<void>;
  unload(): Promise<void>;
  readonly activeLoRA: string | null;
  readonly loaded: boolean;
  readonly modelId: string;
  readonly logicalModelId: string;
  readonly resolvedArtifactVariantId: `sha256:${string}` | null;
  readonly resolutionPolicy: ResolvedDopplerResolutionPolicy;
  readonly manifestHash: string | null;
  readonly persistentCache: DopplerPersistentCacheReceipt | null;
  readonly manifest: unknown;
  readonly deviceInfo: Record<string, unknown> | null;
  readonly supportsSequence: boolean;
  readonly supportsEmbedding: boolean;
  readonly supportsRerank: boolean;
  readonly supportsTranscription: boolean;
  readonly supportsVision: boolean;
  readonly inspect: {
    listPolicies(): DopplerObservationPolicy[];
    generate(
      prompt: string,
      options?: {
        policyId?: string;
        generation?: DopplerGenerateOptions;
        topKSize?: number;
        onEvent?: (event: {
          type: 'inspection-complete';
          receipt: DopplerModelInspectionReceipt;
        }) => void;
      }
    ): Promise<DopplerModelInspectionReceipt>;
  };
  readonly advanced: {
    tokenizeText(text: string): number[];
    prefillKV(prompt: string, options?: DopplerGenerateOptions): Promise<KVCacheSnapshot>;
    resetToSeqLen(seqLen: number): void;
    prefillWithLogits(
      prompt: string | ChatMessage[] | { messages: ChatMessage[] },
      options?: DopplerGenerateOptions
    ): Promise<PrefillResult>;
    prefillWithTokenLogits(
      prompt: PromptInput,
      tokenIds: readonly number[],
      options?: DopplerGenerateOptions
    ): Promise<{
      seqLen: number;
      tokens: number[];
      tokenIds: number[];
      logits: Float32Array;
      logitsByTokenId: Record<number, number>;
      phase?: Record<string, unknown> | null;
    }>;
    prefillWithTokenLogitsFromKV(
      prefix: KVCacheSnapshot,
      prompt: PromptInput,
      tokenIds: readonly number[],
      options?: DopplerGenerateOptions
    ): Promise<{
      seqLen: number;
      prefixTokens: number[];
      tokens: number[];
      tokenIds: number[];
      logits: Float32Array;
      logitsByTokenId: Record<number, number>;
      phase?: Record<string, unknown> | null;
    }>;
    decodeStepLogits(currentIds: number[], options?: DopplerGenerateOptions): Promise<LogitsStepResult>;
    generateWithPrefixKV(
      prefix: KVCacheSnapshot,
      prompt: string,
      options?: DopplerGenerateOptions
    ): AsyncGenerator<string, void, void>;
  };
}

export interface DopplerModelInspectionReceipt {
  schema: 'doppler.model-inspection-receipt/v1';
  policy: DopplerObservationPolicy;
  fingerprint: DopplerComparisonFingerprint;
  outputText: string;
  generatedTokenIds: number[];
  wallTimingMs: number;
  performanceRepresentative: boolean;
  tokens: DopplerInspectionTokenRecord[];
  quality: null | {
    wordSegmentation: 'doppler.word-segmentation/unicode-whitespace-v1';
    aggregation: 'doppler.perplexity/summed-word-surprisal-v1';
    rollingWindow: { unit: 'words' | 'tokens'; size: number };
    words: DopplerWordQualityRecord[];
  };
  generationEvidence: DopplerGenerationEvidence;
}

export declare function assertSupportedGenerationOptions(options?: Record<string, unknown>): void;

export declare function createModelHandle(
  pipeline: InferencePipeline,
  resolved: {
    logicalModelId?: string;
    modelId: string;
    manifestHash?: string | null;
    persistentCache?: DopplerPersistentCacheReceipt | null;
    resolutionPolicy?: DopplerResolutionPolicy | ResolvedDopplerResolutionPolicy | null;
  }
): DopplerModelHandle;
