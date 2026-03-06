import type { ConverterConfigSchema } from '../config/schema/converter.schema.js';
import type { ExecutionContractArtifact } from '../config/execution-contract-check.js';
import type { ExecutionV0GraphContractArtifact } from '../config/execution-v0-graph-contract-check.js';
import type { ManifestRequiredInferenceFieldsArtifact } from '../config/required-inference-fields-contract-check.js';
import type { SavedReportInfo } from '../storage/reports.js';

export interface NodeConvertProgress {
  stage: string | null;
  current: number | null;
  total: number | null;
  message: string | null;
  tensorName?: string | null;
  tensorBytesCurrent?: number | null;
  tensorBytesTotal?: number | null;
}

export interface NodeConvertExecutionConfig {
  workers?: number | null;
  workerCountPolicy?: 'cap' | 'error' | null;
  maxInFlightJobs?: number | null;
  rowChunkRows?: number | null;
  rowChunkMinTensorBytes?: number | null;
  useGpuCast?: boolean | null;
  gpuCastMinTensorBytes?: number | null;
}

export interface ConvertSafetensorsDirectoryOptions {
  /** Directory with safetensors/diffusion assets, or a direct .gguf file path. */
  inputDir: string;
  outputDir?: string | null;
  modelId?: string | null;
  converterConfig?: Partial<ConverterConfigSchema> | null;
  execution?: NodeConvertExecutionConfig | null;
  onProgress?: (progress: NodeConvertProgress) => void;
}

export interface ConvertSafetensorsDirectoryResult {
  manifest: Record<string, unknown>;
  shardCount: number;
  tensorCount: number;
  executionContractArtifact: ExecutionContractArtifact | null;
  executionV0GraphContractArtifact: ExecutionV0GraphContractArtifact | null;
  layerPatternContractArtifact: Record<string, unknown> | null;
  requiredInferenceFieldsArtifact: ManifestRequiredInferenceFieldsArtifact | null;
  report: Record<string, unknown>;
  reportInfo: SavedReportInfo;
  presetId: string;
  modelType: string;
  outputDir: string;
}

export declare function convertSafetensorsDirectory(
  options: ConvertSafetensorsDirectoryOptions
): Promise<ConvertSafetensorsDirectoryResult>;
