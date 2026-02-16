import { InferencePipeline } from '../text.js';
import type { GenerateOptions } from '../text/types.js';

export type DreamStructuredModelType =
  | 'dream_structured'
  | 'dream_intent_posterior_head'
  | 'dream_d1_to2_bridge'
  | 'dream_synthesis'
  | 'dream_energy_compose'
  | 'dream-intent-posterior-head'
  | 'dream-d1-to2-bridge'
  | 'dream-synthesis'
  | 'dream-energy-compose';

export interface DreamInferJSONRequest {
  prompt?: string;
  text?: string;
  nowIso?: string;
  maxTokens?: number;
  temperature?: number;
  maxOutputChars?: number;
  options?: GenerateOptions;
}

export interface DreamInferJSONResult {
  output: Record<string, unknown>;
  rawText: string;
  createdAt: string;
  modelId: string;
  modelHash: unknown;
  promptHash: { alg: 'sha256'; hex: string };
}

export declare class DreamStructuredPipeline extends InferencePipeline {
  inferJSON(request?: DreamInferJSONRequest): Promise<DreamInferJSONResult>;
  infer(request?: DreamInferJSONRequest): Promise<Record<string, unknown>>;
}

export declare function isDreamStructuredModelType(modelType: string | null | undefined): boolean;

export declare function createDreamStructuredPipeline(
  manifest: Record<string, unknown>,
  contexts?: Record<string, unknown>
): Promise<DreamStructuredPipeline>;

