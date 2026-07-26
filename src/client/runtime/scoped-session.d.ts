import type { ChatMessage } from '../../inference/pipelines/text/chat-format.js';
import type { SequenceEncodeOptions, SequenceEncodeResult } from '../../inference/pipelines/text/types.d.ts';
import type {
  DopplerGenerateOptions,
  DopplerGenerationEvidence,
  DopplerModelHandle,
  DopplerModelInspectionReceipt,
} from './model-session.js';
import type { DopplerPersistentCacheReceipt } from './model-source.js';

export type DopplerPromptInput = string | ChatMessage[] | { messages: ChatMessage[] };
export type DopplerCapability =
  | 'generate'
  | 'stream'
  | 'embed'
  | 'sequence'
  | 'inspect'
  | 'lora'
  | 'advanced';
export type DopplerObservationTier = 'always' | 'guided-quality' | 'deep-xray';
export type DopplerObservationPolicyId =
  | 'demo/always-on'
  | 'demo/guided-quality'
  | 'demo/deep-xray';

export interface DopplerScopedGenerateOptions extends DopplerGenerateOptions {
  observe?: DopplerObservationTier | DopplerObservationPolicyId;
  observationPolicy?: DopplerObservationTier | DopplerObservationPolicyId;
}

export interface DopplerGenerationResult {
  schema: 'doppler.generation-result/v1';
  outputText: string;
  content: string;
  tokenIds: number[];
  usage: {
    promptTokens: number | null;
    completionTokens: number;
    totalTokens: number | null;
  };
  observation: {
    policyId: DopplerObservationPolicyId;
    tier: DopplerObservationTier;
    executionClassification: 'representative' | 'observed' | 'deep-diagnostic';
    executionChanged: boolean;
    unavailable: string[];
    deepEvidenceAvailable: boolean | null;
  };
  fingerprint: {
    modelId: string | null;
    manifestHash: string | null;
    tokenizerHash: string | null;
    executionPlanId: string | null;
    kernelPathId: string | null;
    runtimeProfileHash: string | null;
    backendIdentityHash: string | null;
    observationPolicyId: DopplerObservationPolicyId;
  };
  inspectionReceipt: DopplerModelInspectionReceipt | null;
  evidence: DopplerGenerationEvidence;
}

export type DopplerGenerationEvent =
  | {
    schema: 'doppler.generation-event/v1';
    type: 'text-delta';
    text: string;
    observationPolicyId: DopplerObservationPolicyId;
  }
  | {
    schema: 'doppler.generation-event/v1';
    type: 'complete';
    outputText: string;
    observationPolicyId: DopplerObservationPolicyId;
  };

export interface DopplerScopedModelSession {
  readonly schema: 'doppler.scoped-session/v1';
  readonly capabilities: Readonly<Record<DopplerCapability, boolean>>;
  readonly closed: boolean;
  readonly loaded: boolean;
  readonly modelId: string;
  readonly manifestHash: string | null;
  readonly persistentCache: DopplerPersistentCacheReceipt | null;
  readonly manifest: unknown;
  readonly activeLoRA: string | null;
  readonly deviceInfo: Record<string, unknown> | null;
  readonly advanced: DopplerModelHandle['advanced'];
  supports(capability: DopplerCapability): boolean;
  require(capability: DopplerCapability): true;
  generate(input: DopplerPromptInput, options?: DopplerScopedGenerateOptions): Promise<DopplerGenerationResult>;
  stream(input: DopplerPromptInput, options?: DopplerScopedGenerateOptions): AsyncGenerator<DopplerGenerationEvent, void, void>;
  inspect(
    input: string,
    options?: DopplerScopedGenerateOptions & {
      topKSize?: number;
      onEvent?: (event: { type: 'inspection-complete'; receipt: DopplerModelInspectionReceipt }) => void;
    }
  ): Promise<DopplerModelInspectionReceipt>;
  embed(input: string, options?: Record<string, unknown>): Promise<unknown>;
  encodeSequence(sequence: string, options?: SequenceEncodeOptions): Promise<SequenceEncodeResult>;
  loadLoRA(adapter: unknown, options?: Record<string, unknown>): Promise<void>;
  unloadLoRA(): Promise<void>;
  resetGenerationState(): void;
  close(): Promise<void>;
}

export declare const OBSERVATION_POLICIES: Readonly<Record<
  DopplerObservationTier,
  {
    id: DopplerObservationPolicyId;
    tier: DopplerObservationTier;
    executionClassification: 'representative' | 'observed' | 'deep-diagnostic';
  }
>>;

export declare function createScopedModelSession(handle: DopplerModelHandle): DopplerScopedModelSession;
