/**
 * RDRR Manifest Writer
 *
 * @module converter/writer/manifest-writer
 */

import type {
  TensorLocation,
  ShardRecord,
  HashAlgorithm,
  ModelType,
  WeightLayout,
  MoEConfigSchema,
  ComponentGroupSchema,
  TensorMapSchema,
  ConversionInfoSchema,
  RuntimeOptimizationsSchema,
  ManifestInferenceSchema,
  QuantizationInfoSchema,
  WriteResultSchema,
} from './types.js';

export interface ManifestData {
  version: 1;
  modelId: string;
  modelType: ModelType;
  architecture: string;
  quantization: string;
  quantizationInfo?: QuantizationInfoSchema;
  hashAlgorithm: HashAlgorithm;
  config: Record<string, unknown>;
  tokenizer: Record<string, unknown>;
  shards: ShardRecord[];
  groups: Record<string, ComponentGroupSchema>;
  tensorsFile: string;
  tensorCount: number;
  moeConfig: MoEConfigSchema | null;
  totalSize: number;
  defaultWeightLayout?: WeightLayout;
  conversion?: ConversionInfoSchema;
  optimizations?: RuntimeOptimizationsSchema;
  inference?: ManifestInferenceSchema;
}

/**
 * Builds and writes manifest.json and tensors.json.
 */
export declare class ManifestWriter {
  constructor(outputDir: string, hashAlgorithm: HashAlgorithm, modelType: ModelType);

  buildTensorMap(tensorLocations: Map<string, TensorLocation>): TensorMapSchema;
  writeTensorMap(tensorMap: TensorMapSchema): Promise<void>;
  buildGroups(
    groupTensorMap: Map<string, string[]>,
    groupShardMap: Map<string, Set<number>>,
    groupDataMap: Map<string, Uint8Array[]>
  ): Promise<Record<string, ComponentGroupSchema>>;
  buildMoEMapping(
    moeConfig: MoEConfigSchema | null,
    expertShardMap: Map<string, Set<number>>,
    expertTensorMap: Map<string, string[]>,
    expertBytesMap: Map<string, number>,
    sharedExpertIndices: Set<number>
  ): MoEConfigSchema | null;
  writeManifest(manifest: ManifestData): Promise<void>;
  buildResult(manifestPath: string, shards: ShardRecord[], tensorCount: number): WriteResultSchema;
}
