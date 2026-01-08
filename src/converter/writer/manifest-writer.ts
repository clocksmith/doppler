/**
 * RDRR Manifest Writer
 *
 * Generates manifest.json and tensors.json files.
 * Builds component groups with hashes for integrity verification.
 *
 * @module converter/writer/manifest-writer
 */

import { writeFile } from 'fs/promises';
import { join } from 'path';
import {
  getGroupType,
  parseGroupLayerIndex,
  parseGroupExpertIndex,
  sortGroupIds,
  TENSORS_FILENAME,
} from '../../storage/rdrr-format.js';
import { log } from '../../debug/index.js';
import { computeHash } from './utils.js';
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

/** Manifest structure for RDRR v1 format */
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
export class ManifestWriter {
  private outputDir: string;
  private hashAlgorithm: HashAlgorithm;
  private modelType: ModelType;

  constructor(outputDir: string, hashAlgorithm: HashAlgorithm, modelType: ModelType) {
    this.outputDir = outputDir;
    this.hashAlgorithm = hashAlgorithm;
    this.modelType = modelType;
  }

  /**
   * Build external tensors.json from tensor locations.
   */
  buildTensorMap(tensorLocations: Map<string, TensorLocation>): TensorMapSchema {
    const tensorMap: TensorMapSchema = {};
    for (const [name, location] of tensorLocations) {
      tensorMap[name] = {
        shard: location.shardIndex,
        offset: location.offset,
        size: location.size,
        shape: location.shape,
        dtype: location.dtype,
        group: location.group,
      };
      if (location.spans) {
        tensorMap[name].spans = location.spans;
      }
      if (location.layout) {
        tensorMap[name].layout = location.layout;
      }
      if (location.originalShape) {
        tensorMap[name].originalShape = location.originalShape;
      }
    }
    return tensorMap;
  }

  /**
   * Write tensors.json file.
   */
  async writeTensorMap(tensorMap: TensorMapSchema): Promise<void> {
    const tensorsPath = join(this.outputDir, TENSORS_FILENAME);
    await writeFile(tensorsPath, JSON.stringify(tensorMap, null, 2));
  }

  /**
   * Build component groups with hashes from tracked data.
   */
  async buildGroups(
    groupTensorMap: Map<string, string[]>,
    groupShardMap: Map<string, Set<number>>,
    groupDataMap: Map<string, Uint8Array[]>
  ): Promise<Record<string, ComponentGroupSchema>> {
    const groups: Record<string, ComponentGroupSchema> = {};
    const sortedGroupIds = sortGroupIds(Array.from(groupTensorMap.keys()));

    for (const groupId of sortedGroupIds) {
      const tensors = groupTensorMap.get(groupId) || [];
      const shards = Array.from(groupShardMap.get(groupId) || new Set<number>()).sort((a, b) => a - b);
      const dataChunks = groupDataMap.get(groupId) || [];

      // Compute group hash from concatenated tensor data
      const totalSize = dataChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const combined = new Uint8Array(totalSize);
      let offset = 0;
      for (const chunk of dataChunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      const groupHash = await computeHash(combined, this.hashAlgorithm);

      const groupType = getGroupType(groupId, this.modelType);
      const layerIndex = parseGroupLayerIndex(groupId);
      const expertIndex = parseGroupExpertIndex(groupId);

      groups[groupId] = {
        type: groupType,
        version: '1.0.0',
        shards,
        tensors,
        hash: groupHash,
        ...(layerIndex !== undefined && { layerIndex }),
        ...(expertIndex !== undefined && { expertIndex }),
      };
    }

    return groups;
  }

  /**
   * Build MoE expert mapping for legacy compatibility.
   */
  buildMoEMapping(
    moeConfig: MoEConfigSchema | null,
    expertShardMap: Map<string, Set<number>>,
    expertTensorMap: Map<string, string[]>,
    expertBytesMap: Map<string, number>,
    sharedExpertIndices: Set<number>
  ): MoEConfigSchema | null {
    if (!moeConfig || expertShardMap.size === 0) {
      return moeConfig;
    }

    const expertShards: Record<string, number[]> = {};
    const expertTensors: Record<string, string[]> = {};

    for (const [key, shards] of expertShardMap) {
      expertShards[key] = Array.from(shards).sort((a, b) => a - b);
    }

    for (const [key, tensorNames] of expertTensorMap) {
      expertTensors[key] = tensorNames;
    }

    // Calculate average expert size for memory planning
    const expertSizes = Array.from(expertBytesMap.values());
    const expertBytes = expertSizes.length > 0
      ? Math.ceil(expertSizes.reduce((a, b) => a + b, 0) / expertSizes.length)
      : 0;

    const result: MoEConfigSchema = {
      ...moeConfig,
      expertShardMap: expertShards,
      expertTensors,
      expertBytes,
    };

    // Include shared expert indices if any were detected
    if (sharedExpertIndices.size > 0) {
      result.sharedExperts = Array.from(sharedExpertIndices).sort((a, b) => a - b);
      log.verbose('ManifestWriter', `Shared experts: ${result.sharedExperts.join(', ')}`);
    }

    log.verbose('ManifestWriter', `MoE expert mapping: ${expertShardMap.size} experts, ~${(expertBytes / 1024 / 1024).toFixed(1)}MB each`);

    return result;
  }

  /**
   * Write manifest.json file.
   */
  async writeManifest(manifest: ManifestData): Promise<void> {
    const manifestPath = join(this.outputDir, 'manifest.json');
    await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
    log.verbose('ManifestWriter', `Wrote ${Object.keys(manifest.groups).length} component groups`);
  }

  /**
   * Build final write result.
   */
  buildResult(manifestPath: string, shards: ShardRecord[], tensorCount: number): WriteResultSchema {
    const totalSize = shards.reduce((sum, s) => sum + s.size, 0);
    return {
      manifestPath,
      shardCount: shards.length,
      totalSize,
      tensorCount,
    };
  }
}
