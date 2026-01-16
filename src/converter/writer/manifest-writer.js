

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


export class ManifestWriter {
  #outputDir;
  #hashAlgorithm;
  #modelType;

  constructor(outputDir, hashAlgorithm, modelType) {
    this.#outputDir = outputDir;
    this.#hashAlgorithm = hashAlgorithm;
    this.#modelType = modelType;
  }

  
  buildTensorMap(tensorLocations) {
    const tensorMap = {};
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

  
  async writeTensorMap(tensorMap) {
    const tensorsPath = join(this.#outputDir, TENSORS_FILENAME);
    await writeFile(tensorsPath, JSON.stringify(tensorMap, null, 2));
  }

  
  async buildGroups(groupTensorMap, groupShardMap, groupDataMap) {
    const groups = {};
    const sortedGroupIds = sortGroupIds(Array.from(groupTensorMap.keys()));

    for (const groupId of sortedGroupIds) {
      const tensors = groupTensorMap.get(groupId) || [];
      const shards = Array.from(groupShardMap.get(groupId) || new Set()).sort((a, b) => a - b);
      const dataChunks = groupDataMap.get(groupId) || [];

      // Compute group hash from concatenated tensor data
      const totalSize = dataChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const combined = new Uint8Array(totalSize);
      let offset = 0;
      for (const chunk of dataChunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      const groupHash = await computeHash(combined, this.#hashAlgorithm);

      const groupType = getGroupType(groupId, this.#modelType);
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

  
  buildMoEMapping(moeConfig, expertShardMap, expertTensorMap, expertBytesMap, sharedExpertIndices) {
    if (!moeConfig || expertShardMap.size === 0) {
      return moeConfig;
    }

    const expertShards = {};
    const expertTensors = {};

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

    const result = {
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

  
  async writeManifest(manifest) {
    const manifestPath = join(this.#outputDir, 'manifest.json');
    await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
    log.verbose('ManifestWriter', `Wrote ${Object.keys(manifest.groups).length} component groups`);
  }

  
  buildResult(manifestPath, shards, tensorCount) {
    const totalSize = shards.reduce((sum, s) => sum + s.size, 0);
    return {
      manifestPath,
      shardCount: shards.length,
      totalSize,
      tensorCount,
    };
  }
}
