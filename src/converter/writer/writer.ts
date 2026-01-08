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
import { ManifestWriter, type ManifestData } from './manifest-writer.js';
import { TokenizerWriter } from './tokenizer-writer.js';
import { getBytesPerElement, transpose2D } from './utils.js';
import {
  DEFAULT_SHARD_SIZE,
  type TensorMetadata,
  type TensorLocation,
  type TokenizerConfig,
  type HuggingFaceTokenizer,
  type HashAlgorithm,
  type ModelType,
  type WeightLayout,
  type MoEConfigSchema,
  type ConversionInfoSchema,
  type RuntimeOptimizationsSchema,
  type ManifestInferenceSchema,
  type WriterOptionsSchema,
  type WriteResultSchema,
} from './types.js';

/**
 * Main RDRR writer class.
 * Orchestrates tensor writing, manifest generation, and file output.
 */
export class RDRRWriter {
  private outputDir: string;
  private shardSize: number;
  private hashAlgorithm: HashAlgorithm;
  private modelType: ModelType;
  private transposeWeights: boolean;
  private fuseGateUp: boolean;

  private shardWriter: ShardWriter;
  private manifestWriter: ManifestWriter;
  private tokenizerWriter: TokenizerWriter;

  private tensorLocations = new Map<string, TensorLocation>();

  // Component group tracking for v1 format
  private groupTensorMap = new Map<string, string[]>(); // groupId -> [tensor_names]
  private groupShardMap = new Map<string, Set<number>>(); // groupId -> Set<shard_indices>
  private groupDataMap = new Map<string, Uint8Array[]>(); // groupId -> [tensor_data] for hash

  // Expert tensor tracking for MoE models (legacy, also tracked via groups)
  private expertTensorMap = new Map<string, string[]>(); // "0_0" -> [tensor_names]
  private expertShardMap = new Map<string, Set<number>>(); // "0_0" -> Set<shard_indices>
  private expertBytesMap = new Map<string, number>(); // "0_0" -> total_bytes
  private sharedExpertIndices = new Set<number>();

  // FFN gate+up fusion buffering
  // Key: "layer_{idx}" -> { gate?: TensorData, up?: TensorData }
  private ffnFusionBuffer = new Map<string, {
    gate?: { data: Uint8Array; metadata: TensorMetadata; name: string };
    up?: { data: Uint8Array; metadata: TensorMetadata; name: string };
  }>();

  private manifest: ManifestData = {
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

  constructor(outputDir: string, options: WriterOptionsSchema = {}) {
    this.outputDir = outputDir;
    this.shardSize = options.shardSize ?? DEFAULT_SHARD_SIZE;
    this.hashAlgorithm = options.hashAlgorithm ?? 'sha256';
    this.modelType = options.modelType ?? 'transformer';
    this.transposeWeights = options.transposeWeights ?? false;
    this.fuseGateUp = options.fuseGateUp ?? false;

    this.shardWriter = new ShardWriter(outputDir, this.shardSize, this.hashAlgorithm);
    this.manifestWriter = new ManifestWriter(outputDir, this.hashAlgorithm, this.modelType);
    this.tokenizerWriter = new TokenizerWriter(outputDir);

    this.manifest.modelId = options.modelId ?? 'unknown';
    this.manifest.modelType = this.modelType;
    this.manifest.architecture = options.architecture ?? 'llama';
    this.manifest.quantization = options.quantization ?? 'Q4_K_M';
    this.manifest.quantizationInfo = options.quantizationInfo;
    this.manifest.hashAlgorithm = this.hashAlgorithm;
    if (this.transposeWeights) {
      this.manifest.defaultWeightLayout = 'column';
    }
  }

  async init(): Promise<void> {
    await mkdir(this.outputDir, { recursive: true });
    this.shardWriter.startNewShard();
  }

  // ============================================================================
  // Expert Tensor Parsing
  // ============================================================================

  /**
   * Parse expert tensor name and extract layer/expert indices.
   * Returns null for non-expert tensors.
   */
  private parseExpertTensor(name: string): { layerIdx: number; expertIdx: number; isShared?: boolean } | null {
    // Mixtral pattern: model.layers.0.block_sparse_moe.experts.0.w1.weight
    const mixtralMatch = name.match(/layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\./);
    if (mixtralMatch) {
      return { layerIdx: parseInt(mixtralMatch[1], 10), expertIdx: parseInt(mixtralMatch[2], 10) };
    }

    // GPT-OSS pattern: model.layers.0.mlp.experts.0.down_proj.weight
    const gptossMatch = name.match(/layers\.(\d+)\.mlp\.experts\.(\d+)\./);
    if (gptossMatch) {
      return { layerIdx: parseInt(gptossMatch[1], 10), expertIdx: parseInt(gptossMatch[2], 10) };
    }

    // DeepSeek pattern: model.layers.0.mlp.experts.0.gate_proj.weight
    const deepseekMatch = name.match(/layers\.(\d+)\.mlp\.experts\.(\d+)\./);
    if (deepseekMatch) {
      return { layerIdx: parseInt(deepseekMatch[1], 10), expertIdx: parseInt(deepseekMatch[2], 10) };
    }

    // DeepSeek shared expert pattern: model.layers.0.mlp.shared_experts.gate_proj.weight
    const sharedMatch = name.match(/layers\.(\d+)\.mlp\.shared_experts\./);
    if (sharedMatch) {
      // Shared experts use a special index (-1) to indicate they're shared
      return { layerIdx: parseInt(sharedMatch[1], 10), expertIdx: -1, isShared: true };
    }

    // Qwen MoE pattern: model.layers.0.mlp.experts.0.gate_proj.weight
    const qwenMatch = name.match(/layers\.(\d+)\.mlp\.experts\.(\d+)\./);
    if (qwenMatch) {
      return { layerIdx: parseInt(qwenMatch[1], 10), expertIdx: parseInt(qwenMatch[2], 10) };
    }

    // Generic pattern for other MoE architectures
    const genericMatch = name.match(/layers\.(\d+).*experts.*?\.(\d+)\./);
    if (genericMatch) {
      return { layerIdx: parseInt(genericMatch[1], 10), expertIdx: parseInt(genericMatch[2], 10) };
    }

    return null;
  }

  // ============================================================================
  // Weight Transformation
  // ============================================================================

  /**
   * Detect if tensor name is a matmul weight that should be transposed.
   * Matmul weights are 2D tensors used in linear projections (QKV, FFN, etc.)
   */
  private isMatmulWeight(name: string, shape: number[]): boolean {
    // Only transpose 2D tensors
    if (shape.length !== 2) return false;

    // Match common matmul weight patterns
    const matmulPatterns = [
      /\.weight$/,                          // Generic weight suffix
      /q_proj|k_proj|v_proj|o_proj/,        // Attention projections
      /gate_proj|up_proj|down_proj/,        // FFN projections
      /gate\.weight|up\.weight|down\.weight/, // Alternative FFN naming
      /w1\.weight|w2\.weight|w3\.weight/,   // Mixtral FFN naming
      /lm_head/,                            // Language model head
      /embed_tokens/,                       // Token embeddings (transpose for matmul)
    ];

    // Exclude non-matmul tensors
    const excludePatterns = [
      /norm|layernorm|rmsnorm/i,            // Normalization weights (1D)
      /bias$/,                              // Biases (1D)
      /rotary|rope/i,                       // Rotary embeddings
    ];

    for (const pattern of excludePatterns) {
      if (pattern.test(name)) return false;
    }

    for (const pattern of matmulPatterns) {
      if (pattern.test(name)) return true;
    }

    return false;
  }

  // ============================================================================
  // FFN Fusion
  // ============================================================================

  /**
   * Parse FFN projection tensor name and extract layer index and projection type.
   * Returns null if not a gate/up projection.
   */
  private parseFFNProjection(name: string): { layerIdx: number; type: 'gate' | 'up' } | null {
    // Skip expert tensors - they have separate gate/up that shouldn't be fused
    if (name.includes('expert')) return null;

    // Gemma/LLaMA/Mistral pattern: layers.{L}.mlp.gate_proj.weight
    const mlpGateMatch = name.match(/layers\.(\d+)\.mlp\.gate_proj\.weight$/);
    if (mlpGateMatch) {
      return { layerIdx: parseInt(mlpGateMatch[1], 10), type: 'gate' };
    }

    const mlpUpMatch = name.match(/layers\.(\d+)\.mlp\.up_proj\.weight$/);
    if (mlpUpMatch) {
      return { layerIdx: parseInt(mlpUpMatch[1], 10), type: 'up' };
    }

    // GGUF pattern: blk.{L}.ffn_gate.weight
    const ggufGateMatch = name.match(/blk\.(\d+)\.ffn_gate\.weight$/);
    if (ggufGateMatch) {
      return { layerIdx: parseInt(ggufGateMatch[1], 10), type: 'gate' };
    }

    const ggufUpMatch = name.match(/blk\.(\d+)\.ffn_up\.weight$/);
    if (ggufUpMatch) {
      return { layerIdx: parseInt(ggufUpMatch[1], 10), type: 'up' };
    }

    // Alternative naming: w1 = gate, w3 = up (Mixtral/older LLaMA)
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
   * Concatenate two tensors along dimension 0 (rows).
   */
  private concatenateAlongDim0(gate: Uint8Array, up: Uint8Array): Uint8Array {
    const result = new Uint8Array(gate.length + up.length);
    result.set(gate, 0);
    result.set(up, gate.length);
    return result;
  }

  /**
   * Generate the fused tensor name from a gate or up tensor name.
   */
  private getFusedTensorName(name: string): string {
    return name
      .replace(/\.gate_proj\.weight$/, '.gate_up_proj.weight')
      .replace(/\.up_proj\.weight$/, '.gate_up_proj.weight')
      .replace(/\.ffn_gate\.weight$/, '.ffn_gate_up.weight')
      .replace(/\.ffn_up\.weight$/, '.ffn_gate_up.weight')
      .replace(/\.w1\.weight$/, '.w1_w3.weight')
      .replace(/\.w3\.weight$/, '.w1_w3.weight');
  }

  // ============================================================================
  // Group Tracking
  // ============================================================================

  /**
   * Track tensor in its component group for v1 format.
   */
  private trackTensorGroup(name: string, data: Uint8Array, shardIndices: number[]): void {
    const groupId = classifyTensor(name, this.modelType);

    // Track tensor names for this group
    const tensors = this.groupTensorMap.get(groupId) || [];
    tensors.push(name);
    this.groupTensorMap.set(groupId, tensors);

    // Track shard indices for this group
    const shards = this.groupShardMap.get(groupId) || new Set();
    for (const idx of shardIndices) {
      shards.add(idx);
    }
    this.groupShardMap.set(groupId, shards);

    // Track tensor data for group hash computation
    const dataChunks = this.groupDataMap.get(groupId) || [];
    dataChunks.push(data);
    this.groupDataMap.set(groupId, dataChunks);
  }

  /**
   * Track expert tensor for building expertShardMap and expertTensors.
   */
  private trackExpertTensor(name: string, shardIndices: number[], size: number): void {
    const expert = this.parseExpertTensor(name);
    if (!expert) return;

    // Track shared experts separately
    if (expert.isShared) {
      this.sharedExpertIndices.add(expert.expertIdx);
    }

    const key = `${expert.layerIdx}_${expert.expertIdx}`;

    // Track tensor names for this expert
    const tensors = this.expertTensorMap.get(key) || [];
    tensors.push(name);
    this.expertTensorMap.set(key, tensors);

    // Track shard indices for this expert
    const shards = this.expertShardMap.get(key) || new Set();
    for (const idx of shardIndices) {
      shards.add(idx);
    }
    this.expertShardMap.set(key, shards);

    // Track total bytes for this expert
    const currentBytes = this.expertBytesMap.get(key) || 0;
    this.expertBytesMap.set(key, currentBytes + size);
  }

  // ============================================================================
  // Tensor Writing
  // ============================================================================

  async writeTensor(name: string, data: Uint8Array, metadata: TensorMetadata): Promise<TensorLocation> {
    // Check if FFN fusion is enabled and this is a gate/up projection
    if (this.fuseGateUp) {
      const ffnProj = this.parseFFNProjection(name);
      if (ffnProj) {
        // Buffer this tensor for fusion
        const layerKey = `layer_${ffnProj.layerIdx}`;
        if (!this.ffnFusionBuffer.has(layerKey)) {
          this.ffnFusionBuffer.set(layerKey, {});
        }
        const buffer = this.ffnFusionBuffer.get(layerKey)!;

        // Store the tensor
        buffer[ffnProj.type] = { data, metadata, name };

        // Check if we have both gate and up for this layer
        if (buffer.gate && buffer.up) {
          // Validate shapes match
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

          // Concatenate gate and up along dimension 0
          const fusedData = this.concatenateAlongDim0(buffer.gate.data, buffer.up.data);
          const fusedShape = [gateShape[0] + upShape[0], gateShape[1]];
          const fusedName = this.getFusedTensorName(buffer.gate.name);
          const fusedMetadata: TensorMetadata = {
            shape: fusedShape,
            dtype: buffer.gate.metadata.dtype,
          };

          log.verbose('RDRRWriter', `Fusing gate+up for layer ${ffnProj.layerIdx}: ${fusedName} [${fusedShape.join(', ')}]`);

          // Clear the buffer
          this.ffnFusionBuffer.delete(layerKey);

          // Write the fused tensor (recursive call will handle transpose etc.)
          return this.writeTensorInternal(fusedName, fusedData, fusedMetadata);
        }

        // Return a placeholder location (the tensor will be written as part of fused tensor)
        const placeholderLocation: TensorLocation = {
          shardIndex: -1, // Special marker: not written separately
          offset: 0,
          size: 0,
          shape: metadata.shape,
          dtype: metadata.dtype,
        };
        return placeholderLocation;
      }
    }

    // Normal path: write tensor directly
    return this.writeTensorInternal(name, data, metadata);
  }

  /**
   * Internal method to write a tensor to shards.
   */
  private async writeTensorInternal(name: string, data: Uint8Array, metadata: TensorMetadata): Promise<TensorLocation> {
    // Check if this tensor should be transposed
    let writeData = data;
    let writeShape = metadata.shape;
    let layout: WeightLayout | undefined;
    let originalShape: number[] | undefined;

    if (this.transposeWeights && this.isMatmulWeight(name, metadata.shape)) {
      const bytesPerElement = getBytesPerElement(metadata.dtype);
      if (bytesPerElement > 0 && metadata.shape.length === 2) {
        const [rows, cols] = metadata.shape;
        writeData = transpose2D(data, rows, cols, metadata.dtype);
        writeShape = [cols, rows]; // Transposed shape
        layout = 'column';
        originalShape = metadata.shape;
      }
    }

    // Classify tensor into component group
    const groupId = classifyTensor(name, this.modelType);

    // Write to shards
    const spans = await this.shardWriter.writeData(writeData);

    const location: TensorLocation = {
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

    this.tensorLocations.set(name, location);

    // Track tensor in component group for v1 format
    const shardIndices = spans.map(s => s.shardIndex);
    this.trackTensorGroup(name, writeData, shardIndices);

    // Track expert tensor for MoE models (legacy support)
    this.trackExpertTensor(name, shardIndices, data.length);

    return location;
  }

  // ============================================================================
  // Metadata Setters
  // ============================================================================

  setConfig(config: Record<string, unknown>): void {
    this.manifest.config = config;
  }

  setTokenizer(tokenizer: Record<string, unknown>): void {
    this.manifest.tokenizer = tokenizer;
  }

  setMoEConfig(moeConfig: MoEConfigSchema): void {
    this.manifest.moeConfig = moeConfig;
  }

  setConversion(conversion: ConversionInfoSchema): void {
    this.manifest.conversion = conversion;
  }

  setOptimizations(optimizations: RuntimeOptimizationsSchema): void {
    this.manifest.optimizations = optimizations;
  }

  setInference(inference: ManifestInferenceSchema): void {
    this.manifest.inference = inference;
  }

  setMetadata(meta: Record<string, unknown>): void {
    Object.assign(this.manifest, meta);
  }

  // ============================================================================
  // Tokenizer Writing
  // ============================================================================

  async writeTokenizer(tokenizer: TokenizerConfig): Promise<void> {
    const entry = await this.tokenizerWriter.writeTokenizer(tokenizer);
    this.manifest.tokenizer = entry as unknown as Record<string, unknown>;
  }

  async writeHuggingFaceTokenizer(tokenizerJson: HuggingFaceTokenizer): Promise<void> {
    const entry = await this.tokenizerWriter.writeHuggingFaceTokenizer(tokenizerJson);
    if (entry) {
      this.manifest.tokenizer = entry as unknown as Record<string, unknown>;
    }
  }

  // ============================================================================
  // Finalization
  // ============================================================================

  async finalize(): Promise<WriteResultSchema> {
    if (!this.manifest.inference) {
      throw new Error(
        'Manifest inference config is required. ' +
        'Set it via writer.setInference() or pass options.inference to writeRDRR().'
      );
    }

    // Warn about any unpaired gate/up tensors
    if (this.ffnFusionBuffer.size > 0) {
      for (const [layerKey, buffer] of this.ffnFusionBuffer) {
        if (buffer.gate && !buffer.up) {
          log.warn('RDRRWriter', `Layer ${layerKey} has gate_proj but no up_proj - writing unfused`);
          await this.writeTensorInternal(buffer.gate.name, buffer.gate.data, buffer.gate.metadata);
        }
        if (buffer.up && !buffer.gate) {
          log.warn('RDRRWriter', `Layer ${layerKey} has up_proj but no gate_proj - writing unfused`);
          await this.writeTensorInternal(buffer.up.name, buffer.up.data, buffer.up.metadata);
        }
      }
      this.ffnFusionBuffer.clear();
    }

    await this.shardWriter.finalizeShard();

    // Build external tensors.json
    const tensorMap = this.manifestWriter.buildTensorMap(this.tensorLocations);
    await this.manifestWriter.writeTensorMap(tensorMap);

    // Build component groups with hashes
    const groups = await this.manifestWriter.buildGroups(
      this.groupTensorMap,
      this.groupShardMap,
      this.groupDataMap
    );

    // Build shard records for manifest
    const shards = this.shardWriter.finalizedShards.map(s => ({
      index: s.index,
      fileName: s.fileName,
      size: s.size,
      hash: s.hash,
      hashAlgorithm: s.hashAlgorithm,
    }));

    this.manifest.shards = shards;
    this.manifest.groups = groups;
    this.manifest.totalSize = shards.reduce((sum, s) => sum + s.size, 0);
    this.manifest.tensorCount = this.tensorLocations.size;

    // Populate expert shard mapping if MoE model (legacy compatibility)
    this.manifest.moeConfig = this.manifestWriter.buildMoEMapping(
      this.manifest.moeConfig,
      this.expertShardMap,
      this.expertTensorMap,
      this.expertBytesMap,
      this.sharedExpertIndices
    );

    // Write manifest.json
    await this.manifestWriter.writeManifest(this.manifest);

    const manifestPath = join(this.outputDir, 'manifest.json');
    return this.manifestWriter.buildResult(manifestPath, shards, this.manifest.tensorCount);
  }

  async cleanup(): Promise<void> {
    try {
      await rm(this.outputDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}
