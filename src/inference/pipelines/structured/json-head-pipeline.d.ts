import { InferencePipeline } from '../text.js';
import type { GenerateOptions } from '../text/types.js';

export type StructuredJsonHeadModelType =
  | 'structured_json_head'
  | 'structured-json-head'
  | 'dream_structured'
  | 'dream_intent_posterior_head'
  | 'dream_d1_to2_bridge'
  | 'dream_synthesis'
  | 'dream_energy_compose'
  | 'dream-intent-posterior-head'
  | 'dream-d1-to2-bridge'
  | 'dream-synthesis'
  | 'dream-energy-compose';

export interface StructuredJsonHeadInferJSONRequest {
  prompt?: string;
  text?: string;
  nowIso?: string;
  maxTokens?: number;
  temperature?: number;
  maxOutputChars?: number;
  options?: GenerateOptions;
}

export interface StructuredJsonHeadInferJSONResult {
  output: Record<string, unknown>;
  rawText: string;
  createdAt: string;
  modelId: string;
  modelHash: unknown;
  promptHash: { alg: 'sha256'; hex: string };
}

export declare class StructuredJsonHeadPipeline extends InferencePipeline {
  inferJSON(request?: StructuredJsonHeadInferJSONRequest): Promise<StructuredJsonHeadInferJSONResult>;
  infer(request?: StructuredJsonHeadInferJSONRequest): Promise<Record<string, unknown>>;
}

export declare function isStructuredJsonHeadModelType(modelType: string | null | undefined): boolean;

export declare function createStructuredJsonHeadPipeline(
  manifest: Record<string, unknown>,
  contexts?: Record<string, unknown>
): Promise<StructuredJsonHeadPipeline>;

export type DreamStructuredModelType = StructuredJsonHeadModelType;

export type DreamInferJSONRequest = StructuredJsonHeadInferJSONRequest;

export type DreamInferJSONResult = StructuredJsonHeadInferJSONResult;

export declare class DreamStructuredPipeline extends StructuredJsonHeadPipeline {}

export declare const isDreamStructuredModelType: typeof isStructuredJsonHeadModelType;

export declare const createDreamStructuredPipeline: typeof createStructuredJsonHeadPipeline;
