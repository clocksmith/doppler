import type { ConverterConfigSchema } from '../config/schema/converter.schema.js';

export interface ConversionPlanInputTensor {
  name: string;
  dtype?: string | null;
}

export interface ResolveConversionPlanOptions {
  rawConfig: Record<string, unknown> | null | undefined;
  tensors?: ConversionPlanInputTensor[] | null;
  tensorNames?: string[] | null;
  converterConfig: ConverterConfigSchema;
  sourceQuantization?: string | null;
  modelKind?: 'transformer' | 'diffusion';
  architectureHint?: string | null;
  architecture?: string | null;
  architectureConfig?: { headDim?: number | null } | null;
  headDim?: number | null;
  headDimErrorMessage?: string | null;
  presetOverride?: string | null;
  includePresetOverrideHint?: boolean;
}

export interface ConversionPlanResult {
  modelType: string;
  presetId: string;
  preset: Record<string, unknown> | null;
  sourceQuantization: string;
  quantizationInfo: Record<string, unknown>;
  manifestQuantization: string;
  manifestInference: Record<string, unknown>;
  headDim?: number;
}

export interface ValidateKernelPathContext {
  presetId?: string | null;
  quantizationInfo?: {
    weights?: string | null;
    compute?: string | null;
    layout?: string | null;
  } | null;
}

export interface ResolveConvertedModelIdOptions {
  explicitModelId?: string | null;
  converterConfig?: ConverterConfigSchema | null;
  detectedModelId?: string | null;
  fallbackModelId?: string | null;
  quantizationInfo?: { variantTag?: string | null } | null;
  sanitizeOnly?: boolean;
}

export declare function inferSourceWeightQuantization(
  tensors: ConversionPlanInputTensor[] | null | undefined
): string;

export declare function validateDefaultKernelPath(
  inference: { defaultKernelPath?: string | null } | null | undefined,
  context?: ValidateKernelPathContext
): void;

export declare function resolveConversionPlan(options: ResolveConversionPlanOptions): ConversionPlanResult;

export declare function resolveConvertedModelId(options: ResolveConvertedModelIdOptions): string | null;
