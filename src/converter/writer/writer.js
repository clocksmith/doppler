/**
 * RDRR Writer - Main Orchestration Class
 *
 * Coordinates shard writing, manifest generation, and tokenizer bundling.
 * Handles tensor transformations (transpose, FFN fusion) and expert tracking.
 *
 * @module converter/writer/writer
 */

import { mkdir, rm } from 'fs/promises';
import { join } from 'path';
import { classifyTensor, TENSORS_FILENAME } from '../../storage/rdrr-format.js';
import { log } from '../../debug/index.js';
import { ShardWriter } from './shard-writer.js';
import { ManifestWriter } from './manifest-writer.js';
import { TokenizerWriter } from './tokenizer-writer.js';
import { getBytesPerElement, transpose2D } from './utils.js';
import { DEFAULT_SHARD_SIZE } from './types.js';

/**
 * Main RDRR writer class.
 * Orchestrates tensor writing, manifest generation, and file output.
 */
export class RDRRWriter {
  #outputDir;
  #shardSize;
  #hashAlgorithm;
  #modelType;
  #transposeWeights;
  #fuseGateUp;

  #shardWriter;
  #manifestWriter;
  #tokenizerWriter;

  #tensorLocations = new Map();

  // Component group tracking
  #groupTensorMap = new Map();
  #groupShardMap = new Map();
  #groupDataMap = new Map();

  // Expert tensor tracking
  #expertTensorMap = new Map();
  #expertShardMap = new Map();
  #expertBytesMap = new Map();
  #sharedExpertIndices = new Set();

  // FFN gate+up fusion buffering
  #ffnFusionBuffer = new Map();

  #manifest = {
    version: 1,
    modelId: 'unknown',
    modelType: 'transformer',
    architecture: 'llama',
    quantization: 'Q4_K_M',
    quantizationInfo: undefined,
    hashAlgorithm: 'sha256',
    config: {},
    tokenizer: {},
    shards: [],
    groups: {},
    tensorsFile: TENSORS_FILENAME,
    tensorCount: 0,
    moeConfig: null,
    totalSize: 0,
    defaultWeightLayout: undefined,
    conversion: undefined,
    optimizations: undefined,
    inference: undefined,
  };

  constructor(outputDir, options = {}) {
    this.#outputDir = outputDir;
    this.#shardSize = options.shardSize ?? DEFAULT_SHARD_SIZE;
    this.#hashAlgorithm = options.hashAlgorithm ?? 'sha256';
    this.#modelType = options.modelType ?? 'transformer';
    this.#transposeWeights = options.transposeWeights ?? false;
    this.#fuseGateUp = options.fuseGateUp ?? false;

    this.#shardWriter = new ShardWriter(outputDir, this.#shardSize, this.#hashAlgorithm);
    this.#manifestWriter = new ManifestWriter(outputDir, this.#hashAlgorithm, this.#modelType);
    this.#tokenizerWriter = new TokenizerWriter(outputDir);

    this.#manifest.modelId = options.modelId ?? 'unknown';
    this.#manifest.modelType = this.#modelType;
    this.#manifest.architecture = options.architecture ?? 'llama';
    this.#manifest.quantization = options.quantization ?? 'Q4_K_M';
    this.#manifest.quantizationInfo = options.quantizationInfo;
    this.#manifest.hashAlgorithm = this.#hashAlgorithm;
    if (this.#transposeWeights) {
      this.#manifest.defaultWeightLayout = 'column';
    }
  }

  async init() {
    await mkdir(this.#outputDir, { recursive: true });
    this.#shardWriter.startNewShard();
  }

  /**
   * Parse expert tensor name and extract layer/expert indices.
   */
  #parseExpertTensor(name) {
    const mixtralMatch = name.match(/layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\./);
    if (mixtralMatch) {
      return { layerIdx: parseInt(mixtralMatch[1], 10), expertIdx: parseInt(mixtralMatch[2], 10) };
    }

    const gptossMatch = name.match(/layers\.(\d+)\.mlp\.experts\.(\d+)\./);
    if (gptossMatch) {
      return { layerIdx: parseInt(gptossMatch[1], 10), expertIdx: parseInt(gptossMatch[2], 10) };
    }

    const deepseekMatch = name.match(/layers\.(\d+)\.mlp\.experts\.(\d+)\./);
    if (deepseekMatch) {
      return { layerIdx: parseInt(deepseekMatch[1], 10), expertIdx: parseInt(deepseekMatch[2], 10) };
    }

    const sharedMatch = name.match(/layers\.(\d+)\.mlp\.shared_experts\./);
    if (sharedMatch) {
      return { layerIdx: parseInt(sharedMatch[1], 10), expertIdx: -1, isShared: true };
    }

    const qwenMatch = name.match(/layers\.(\d+)\.mlp\.experts\.(\d+)\./);
    if (qwenMatch) {
      return { layerIdx: parseInt(qwenMatch[1], 10), expertIdx: parseInt(qwenMatch[2], 10) };
    }

    const genericMatch = name.match(/layers\.(\d+).*experts.*?\.(\d+)\./);
    if (genericMatch) {
      return { layerIdx: parseInt(genericMatch[1], 10), expertIdx: parseInt(genericMatch[2], 10) };
    }

    return null;
  }

  /**
   * Detect if tensor name is a matmul weight that should be transposed.
   */
  #isMatmulWeight(name, shape) {
    if (shape.length !== 2) return false;

    const matmulPatterns = [
      /\.weight$/,
      /q_proj|k_proj|v_proj|o_proj/,
      /gate_proj|up_proj|down_proj/,
      /gate\.weight|up\.weight|down\.weight/,
      /w1\.weight|w2\.weight|w3\.weight/,
      /lm_head/,
      /embed_tokens/,
    ];

    const excludePatterns = [
      /norm|layernorm|rmsnorm/i,
      /bias$/,
      /rotary|rope/i,
    ];

    for (const pattern of excludePatterns) {
      if (pattern.test(name)) return false;
    }

    for (const pattern of matmulPatterns) {
      if (pattern.test(name)) return true;
    }

    return false;
  }

  /**
   * Parse FFN projection tensor name.
   */
  #parseFFNProjection(name) {
    if (name.includes('expert')) return null;

    const mlpGateMatch = name.match(/layers\.(\d+)\.mlp\.gate_proj\.weight$/);
    if (mlpGateMatch) {
      return { layerIdx: parseInt(mlpGateMatch[1], 10), type: 'gate' };
    }

    const mlpUpMatch = name.match(/layers\.(\d+)\.mlp\.up_proj\.weight$/);
    if (mlpUpMatch) {
      return { layerIdx: parseInt(mlpUpMatch[1], 10), type: 'up' };
    }

    const ggufGateMatch = name.match(/blk\.(\d+)\.ffn_gate\.weight$/);
    if (ggufGateMatch) {
      return { layerIdx: parseInt(ggufGateMatch[1], 10), type: 'gate' };
    }

    const ggufUpMatch = name.match(/blk\.(\d+)\.ffn_up\.weight$/);
    if (ggufUpMatch) {
      return { layerIdx: parseInt(ggufUpMatch[1], 10), type: 'up' };
    }

    const w1Match = name.match(/layers\.(\d+)\.(?:feed_forward|mlp)\.w1\.weight$/);
    if (w1Match) {
      return { layerIdx: parseInt(w1Match[1], 10), type: 'gate' };
    }

    const w3Match = name.match(/layers\.(\d+)\.(?:feed_forward|mlp)\.w3\.weight$/);
    if (w3Match) {
      return { layerIdx: parseInt(w3Match[1], 10), type: 'up' };
    }

    return null;
  }

  /**
   * Concatenate two tensors along dimension 0.
   */
  #concatenateAlongDim0(gate, up) {
    const result = new Uint8Array(gate.length + up.length);
    result.set(gate, 0);
    result.set(up, gate.length);
    return result;
  }

  /**
   * Generate the fused tensor name.
   */
  #getFusedTensorName(name) {
    return name
      .replace(/\.gate_proj\.weight$/, '.gate_up_proj.weight')
      .replace(/\.up_proj\.weight$/, '.gate_up_proj.weight')
      .replace(/\.ffn_gate\.weight$/, '.ffn_gate_up.weight')
      .replace(/\.ffn_up\.weight$/, '.ffn_gate_up.weight')
      .replace(/\.w1\.weight$/, '.w1_w3.weight')
      .replace(/\.w3\.weight$/, '.w1_w3.weight');
  }

  /**
   * Track tensor in component group.
   */
  #trackTensorGroup(name, data, shardIndices) {
    const groupId = classifyTensor(name, this.#modelType);

    const tensors = this.#groupTensorMap.get(groupId) || [];
    tensors.push(name);
    this.#groupTensorMap.set(groupId, tensors);

    const shards = this.#groupShardMap.get(groupId) || new Set();
    for (const idx of shardIndices) {
      shards.add(idx);
    }
    this.#groupShardMap.set(groupId, shards);

    const dataChunks = this.#groupDataMap.get(groupId) || [];
    dataChunks.push(data);
    this.#groupDataMap.set(groupId, dataChunks);
  }

  /**
   * Track expert tensor.
   */
  #trackExpertTensor(name, shardIndices, size) {
    const expert = this.#parseExpertTensor(name);
    if (!expert) return;

    if (expert.isShared) {
      this.#sharedExpertIndices.add(expert.expertIdx);
    }

    const key = `${expert.layerIdx}_${expert.expertIdx}`;

    const tensors = this.#expertTensorMap.get(key) || [];
    tensors.push(name);
    this.#expertTensorMap.set(key, tensors);

    const shards = this.#expertShardMap.get(key) || new Set();
    for (const idx of shardIndices) {
      shards.add(idx);
    }
    this.#expertShardMap.set(key, shards);

    const currentBytes = this.#expertBytesMap.get(key) || 0;
    this.#expertBytesMap.set(key, currentBytes + size);
  }

  async writeTensor(name, data, metadata) {
    if (this.#fuseGateUp) {
      const ffnProj = this.#parseFFNProjection(name);
      if (ffnProj) {
        const layerKey = `layer_${ffnProj.layerIdx}`;
        if (!this.#ffnFusionBuffer.has(layerKey)) {
          this.#ffnFusionBuffer.set(layerKey, {});
        }
        const buffer = this.#ffnFusionBuffer.get(layerKey);

        buffer[ffnProj.type] = { data, metadata, name };

        if (buffer.gate && buffer.up) {
          const gateShape = buffer.gate.metadata.shape;
          const upShape = buffer.up.metadata.shape;
          if (gateShape[1] !== upShape[1]) {
            throw new Error(`FFN fusion shape mismatch at layer ${ffnProj.layerIdx}: ` +
              `gate shape ${gateShape}, up shape ${upShape}`);
          }
          if (buffer.gate.metadata.dtype !== buffer.up.metadata.dtype) {
            throw new Error(`FFN fusion dtype mismatch at layer ${ffnProj.layerIdx}: ` +
              `gate dtype ${buffer.gate.metadata.dtype}, up dtype ${buffer.up.metadata.dtype}`);
          }

          const fusedData = this.#concatenateAlongDim0(buffer.gate.data, buffer.up.data);
          const fusedShape = [gateShape[0] + upShape[0], gateShape[1]];
          const fusedName = this.#getFusedTensorName(buffer.gate.name);
          const fusedMetadata = {
            shape: fusedShape,
            dtype: buffer.gate.metadata.dtype,
          };

          log.verbose('RDRRWriter', `Fusing gate+up for layer ${ffnProj.layerIdx}: ${fusedName} [${fusedShape.join(', ')}]`);

          this.#ffnFusionBuffer.delete(layerKey);

          return this.#writeTensorInternal(fusedName, fusedData, fusedMetadata);
        }

        const placeholderLocation = {
          shardIndex: -1,
          offset: 0,
          size: 0,
          shape: metadata.shape,
          dtype: metadata.dtype,
        };
        return placeholderLocation;
      }
    }

    return this.#writeTensorInternal(name, data, metadata);
  }

  async #writeTensorInternal(name, data, metadata) {
    let writeData = data;
    let writeShape = metadata.shape;
    let layout;
    let originalShape;

    if (this.#transposeWeights && this.#isMatmulWeight(name, metadata.shape)) {
      const bytesPerElement = getBytesPerElement(metadata.dtype);
      if (bytesPerElement > 0 && metadata.shape.length === 2) {
        const [rows, cols] = metadata.shape;
        writeData = transpose2D(data, rows, cols, metadata.dtype);
        writeShape = [cols, rows];
        layout = 'column';
        originalShape = metadata.shape;
      }
    }

    const groupId = classifyTensor(name, this.#modelType);

    const spans = await this.#shardWriter.writeData(writeData);

    const location = {
      shardIndex: spans[0].shardIndex,
      offset: spans[0].offset,
      size: writeData.length,
      shape: writeShape,
      dtype: metadata.dtype,
      layout,
      originalShape,
      group: groupId,
    };

    if (spans.length > 1) {
      location.spans = spans;
    }

    this.#tensorLocations.set(name, location);

    const shardIndices = spans.map(s => s.shardIndex);
    this.#trackTensorGroup(name, writeData, shardIndices);
    this.#trackExpertTensor(name, shardIndices, data.length);

    return location;
  }

  setConfig(config) {
    this.#manifest.config = config;
  }

  setTokenizer(tokenizer) {
    this.#manifest.tokenizer = tokenizer;
  }

  setMoEConfig(moeConfig) {
    this.#manifest.moeConfig = moeConfig;
  }

  setConversion(conversion) {
    this.#manifest.conversion = conversion;
  }

  setOptimizations(optimizations) {
    this.#manifest.optimizations = optimizations;
  }

  setInference(inference) {
    this.#manifest.inference = inference;
  }

  setMetadata(meta) {
    Object.assign(this.#manifest, meta);
  }

  async writeTokenizer(tokenizer) {
    const entry = await this.#tokenizerWriter.writeTokenizer(tokenizer);
    this.#manifest.tokenizer = entry;
  }

  async writeHuggingFaceTokenizer(tokenizerJson) {
    const entry = await this.#tokenizerWriter.writeHuggingFaceTokenizer(tokenizerJson);
    if (entry) {
      this.#manifest.tokenizer = entry;
    }
  }

  async finalize() {
    if (!this.#manifest.inference) {
      throw new Error(
        'Manifest inference config is required. ' +
        'Set it via writer.setInference() or pass options.inference to writeRDRR().'
      );
    }

    if (this.#ffnFusionBuffer.size > 0) {
      for (const [layerKey, buffer] of this.#ffnFusionBuffer) {
        if (buffer.gate && !buffer.up) {
          log.warn('RDRRWriter', `Layer ${layerKey} has gate_proj but no up_proj - writing unfused`);
          await this.#writeTensorInternal(buffer.gate.name, buffer.gate.data, buffer.gate.metadata);
        }
        if (buffer.up && !buffer.gate) {
          log.warn('RDRRWriter', `Layer ${layerKey} has up_proj but no gate_proj - writing unfused`);
          await this.#writeTensorInternal(buffer.up.name, buffer.up.data, buffer.up.metadata);
        }
      }
      this.#ffnFusionBuffer.clear();
    }

    await this.#shardWriter.finalizeShard();

    const tensorMap = this.#manifestWriter.buildTensorMap(this.#tensorLocations);
    await this.#manifestWriter.writeTensorMap(tensorMap);

    const groups = await this.#manifestWriter.buildGroups(
      this.#groupTensorMap,
      this.#groupShardMap,
      this.#groupDataMap
    );

    const shards = this.#shardWriter.finalizedShards.map(s => ({
      index: s.index,
      fileName: s.fileName,
      size: s.size,
      hash: s.hash,
      hashAlgorithm: s.hashAlgorithm,
    }));

    this.#manifest.shards = shards;
    this.#manifest.groups = groups;
    this.#manifest.totalSize = shards.reduce((sum, s) => sum + s.size, 0);
    this.#manifest.tensorCount = this.#tensorLocations.size;

    this.#manifest.moeConfig = this.#manifestWriter.buildMoEMapping(
      this.#manifest.moeConfig,
      this.#expertShardMap,
      this.#expertTensorMap,
      this.#expertBytesMap,
      this.#sharedExpertIndices
    );

    await this.#manifestWriter.writeManifest(this.#manifest);

    const manifestPath = join(this.#outputDir, 'manifest.json');
    return this.#manifestWriter.buildResult(manifestPath, shards, this.#manifest.tensorCount);
  }

  async cleanup() {
    try {
      await rm(this.#outputDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}
