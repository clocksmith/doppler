/**
 * .rdrr Model Format Writer
 * Writes models with manifest.json + 64MB shards with hash verification.
 */

import { writeFile, mkdir, rm } from 'fs/promises';
import { join } from 'path';
import { createHash } from 'crypto';
import type {
  HashAlgorithm,
  MoEConfig,
  WeightLayout,
  ConversionInfo,
  RuntimeOptimizations,
} from '../storage/rdrr-format.js';

const DEFAULT_SHARD_SIZE = 64 * 1024 * 1024;
const ALIGNMENT = 4096;

export interface WriterOptions {
  shardSize?: number;
  hashAlgorithm?: HashAlgorithm;
  modelId?: string;
  modelType?: string;
  architecture?: string;
  quantization?: string;
  /** Pre-transpose matmul weights to column-major for faster GPU access (default: false) */
  transposeWeights?: boolean;
  /** Fuse gate_proj + up_proj into gate_up_proj for 2-pass FFN (default: false) */
  fuseGateUp?: boolean;
}

export interface TensorMetadata {
  shape: number[];
  dtype: string;
}

export interface TensorLocation extends TensorMetadata {
  shardIndex: number;
  offset: number;
  size: number;
  spans?: Array<{ shardIndex: number; offset: number; size: number }>;
  /** Weight storage layout: 'row' (default) or 'column' (pre-transposed) */
  layout?: WeightLayout;
  /** Original shape before transpose (if layout is 'column') */
  originalShape?: number[];
}

export interface ShardData {
  index: number;
  data: Uint8Array[];
  size: number;
}

export interface ShardRecord {
  index: number;
  fileName: string;
  size: number;
  hash: string;
  hashAlgorithm: HashAlgorithm;
}

export interface WriteResult {
  manifestPath: string;
  shardCount: number;
  totalSize: number;
  tensorCount: number;
}

export interface TokenizerConfig {
  model?: string;
  tokens?: string[];
  merges?: string[];
  scores?: number[];
  tokenTypes?: number[];
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
  unkTokenId?: number;
  sepTokenId?: number;
  clsTokenId?: number;
  maskTokenId?: number;
  addBosToken?: boolean;
  addEosToken?: boolean;
  addSpacePrefix?: boolean;
}

export interface HuggingFaceTokenizer {
  model?: {
    type?: string;
    vocab?: Record<string, number> | Array<[string, number]>;
  };
}

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
}

export interface ModelInfo {
  modelName?: string;
  architecture?: string;
  quantization?: string;
  config?: Record<string, unknown>;
  tokenizer?: TokenizerConfig;
  tokenizerConfig?: TokenizerConfig;
  tokenizerJson?: HuggingFaceTokenizer;
  tensors: TensorInfo[];
}

export interface ProgressEvent {
  stage: 'writing' | 'complete';
  current?: number;
  total?: number;
  tensorName?: string;
  manifestPath?: string;
  shardCount?: number;
  totalSize?: number;
  tensorCount?: number;
}

async function computeHash(data: Uint8Array): Promise<string> {
  try {
    // blake3 package may not have types, use dynamic import
    // @ts-expect-error - blake3 has no type declarations
    const blake3Module = await import('blake3');
    return blake3Module.blake3(data).toString('hex');
  } catch {
    const hash = createHash('sha256');
    hash.update(data);
    return hash.digest('hex');
  }
}

function alignOffset(offset: number, alignment = ALIGNMENT): number {
  const remainder = offset % alignment;
  return remainder === 0 ? offset : offset + (alignment - remainder);
}

function createPadding(size: number): Uint8Array {
  return new Uint8Array(size);
}

export class RDRRWriter {
  private outputDir: string;
  private shardSize: number;
  private hashAlgorithm: HashAlgorithm;
  private transposeWeights: boolean;
  private fuseGateUp: boolean;
  private shards: ShardRecord[] = [];
  private currentShard: ShardData | null = null;
  private currentShardIndex = 0;
  private currentShardOffset = 0;
  private tensorLocations = new Map<string, TensorLocation>();

  // Expert tensor tracking for MoE models
  private expertTensorMap = new Map<string, string[]>(); // "0_0" -> [tensor_names]
  private expertShardMap = new Map<string, Set<number>>(); // "0_0" -> Set<shard_indices>
  private expertBytesMap = new Map<string, number>(); // "0_0" -> total_bytes

  // FFN gate+up fusion buffering
  // Key: "layer_{idx}" -> { gate?: TensorData, up?: TensorData }
  private ffnFusionBuffer = new Map<string, {
    gate?: { data: Uint8Array; metadata: TensorMetadata; name: string };
    up?: { data: Uint8Array; metadata: TensorMetadata; name: string };
  }>();

  private manifest = {
    version: '1.0' as const,
    modelId: 'unknown',
    modelType: 'transformer',
    architecture: 'llama',
    quantization: 'Q4_K_M',
    hashAlgorithm: 'sha256' as HashAlgorithm,
    config: {} as Record<string, unknown>,
    tokenizer: {} as Record<string, unknown>,
    shards: [] as ShardRecord[],
    tensors: {} as Record<string, TensorLocation>,
    moeConfig: null as MoEConfig | null,
    totalSize: 0,
    tensorCount: 0,
    defaultWeightLayout: undefined as WeightLayout | undefined,
    conversion: undefined as ConversionInfo | undefined,
    optimizations: undefined as RuntimeOptimizations | undefined,
  };

  constructor(outputDir: string, options: WriterOptions = {}) {
    this.outputDir = outputDir;
    this.shardSize = options.shardSize ?? DEFAULT_SHARD_SIZE;
    this.hashAlgorithm = options.hashAlgorithm ?? 'sha256';
    this.transposeWeights = options.transposeWeights ?? false;
    this.fuseGateUp = options.fuseGateUp ?? false;

    this.manifest.modelId = options.modelId ?? 'unknown';
    this.manifest.modelType = options.modelType ?? 'transformer';
    this.manifest.architecture = options.architecture ?? 'llama';
    this.manifest.quantization = options.quantization ?? 'Q4_K_M';
    this.manifest.hashAlgorithm = this.hashAlgorithm;
    if (this.transposeWeights) {
      this.manifest.defaultWeightLayout = 'column';
    }
  }

  async init(): Promise<void> {
    await mkdir(this.outputDir, { recursive: true });
    this.startNewShard();
  }

  /**
   * Detect if tensor name is an expert weight and extract layer/expert indices.
   * Patterns supported:
   * - Mixtral: layers.{L}.block_sparse_moe.experts.{E}.w{1,2,3}.weight
   * - GPT-OSS: model.layers.{L}.mlp.experts.{E}.*
   * - Generic: *.layers.{L}.*experts*.{E}.*
   */
  /**
   * Parse expert tensor name and extract layer/expert indices
   * Returns null for non-expert tensors
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

  /**
   * Transpose a 2D tensor from [rows, cols] to [cols, rows].
   * Works with any element size by operating on bytes.
   */
  private transpose2D(data: Uint8Array, rows: number, cols: number, dtype: string): Uint8Array {
    const bytesPerElement = this.getBytesPerElement(dtype);
    const rowBytes = cols * bytesPerElement;
    const colBytes = rows * bytesPerElement;
    const result = new Uint8Array(data.length);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const srcOffset = (r * cols + c) * bytesPerElement;
        const dstOffset = (c * rows + r) * bytesPerElement;
        for (let b = 0; b < bytesPerElement; b++) {
          result[dstOffset + b] = data[srcOffset + b];
        }
      }
    }

    return result;
  }

  private getBytesPerElement(dtype: string): number {
    const dtypeLower = dtype.toLowerCase();
    if (dtypeLower === 'f32' || dtypeLower === 'float32') return 4;
    if (dtypeLower === 'f16' || dtypeLower === 'float16') return 2;
    if (dtypeLower === 'bf16' || dtypeLower === 'bfloat16') return 2;
    if (dtypeLower === 'i32' || dtypeLower === 'int32') return 4;
    if (dtypeLower === 'i16' || dtypeLower === 'int16') return 2;
    if (dtypeLower === 'i8' || dtypeLower === 'int8') return 1;
    // Q4_K, Q8_0, etc. - these are block-quantized, don't transpose
    return 0; // 0 indicates no transpose possible
  }

  /**
   * Parse FFN projection tensor name and extract layer index and projection type.
   * Returns null if not a gate/up projection.
   * Supported patterns:
   * - layers.{L}.mlp.gate_proj.weight
   * - layers.{L}.mlp.up_proj.weight
   * - model.layers.{L}.mlp.gate_proj.weight
   * - layers.{L}.feed_forward.w1.weight (gate)
   * - layers.{L}.feed_forward.w3.weight (up)
   * - blk.{L}.ffn_gate.weight
   * - blk.{L}.ffn_up.weight
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
   * gate: [intermediateSize, hiddenSize]
   * up:   [intermediateSize, hiddenSize]
   * result: [intermediateSize*2, hiddenSize] with gate first, then up
   */
  private concatenateAlongDim0(gate: Uint8Array, up: Uint8Array): Uint8Array {
    const result = new Uint8Array(gate.length + up.length);
    result.set(gate, 0);
    result.set(up, gate.length);
    return result;
  }

  /**
   * Generate the fused tensor name from a gate or up tensor name.
   * e.g., layers.0.mlp.gate_proj.weight -> layers.0.mlp.gate_up_proj.weight
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

  // Track shared expert indices for the model
  private sharedExpertIndices = new Set<number>();

  /**
   * Track expert tensor for building expertShardMap and expertTensors
   */
  private trackExpertTensor(name: string, shardIndices: number[], size: number): void {
    const expert = this.parseExpertTensor(name);
    if (!expert) return;

    // Track shared experts separately
    if (expert.isShared) {
      this.sharedExpertIndices.add(expert.expertIdx);
      // Still track shared experts in the shard map for loading
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

  private startNewShard(): void {
    if (this.currentShard && this.currentShardOffset > 0) {
      this.finalizeShard();
    }

    this.currentShard = {
      index: this.currentShardIndex,
      data: [],
      size: 0,
    };
  }

  private async finalizeShard(): Promise<void> {
    if (!this.currentShard || this.currentShard.size === 0) {
      return;
    }

    const totalSize = this.currentShard.data.reduce((sum, chunk) => sum + chunk.length, 0);
    const shardData = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of this.currentShard.data) {
      shardData.set(chunk, offset);
      offset += chunk.length;
    }

    const hash = await computeHash(shardData);
    const shardFileName = `shard_${String(this.currentShardIndex).padStart(5, '0')}.bin`;
    const shardPath = join(this.outputDir, shardFileName);
    await writeFile(shardPath, shardData);

    this.shards.push({
      index: this.currentShardIndex,
      fileName: shardFileName,
      size: totalSize,
      hash,
      hashAlgorithm: this.hashAlgorithm,
    });

    this.currentShardIndex++;
    this.currentShardOffset = 0;
    this.currentShard = null;
  }

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

          console.log(`[RDRRWriter] Fusing gate+up for layer ${ffnProj.layerIdx}: ${fusedName} [${fusedShape.join(', ')}]`);

          // Clear the buffer
          this.ffnFusionBuffer.delete(layerKey);

          // Write the fused tensor (recursive call will handle transpose etc.)
          return this.writeTensorInternal(fusedName, fusedData, fusedMetadata);
        }

        // Return a placeholder location (the tensor will be written as part of fused tensor)
        // We use a special marker that the loader will understand
        const placeholderLocation: TensorLocation = {
          shardIndex: -1, // Special marker: not written separately
          offset: 0,
          size: 0,
          shape: metadata.shape,
          dtype: metadata.dtype,
        };
        // Don't add to tensorLocations - the fused tensor will be the canonical one
        return placeholderLocation;
      }
    }

    // Normal path: write tensor directly
    return this.writeTensorInternal(name, data, metadata);
  }

  /**
   * Internal method to write a tensor to shards.
   * Handles transpose, alignment, and shard splitting.
   */
  private async writeTensorInternal(name: string, data: Uint8Array, metadata: TensorMetadata): Promise<TensorLocation> {
    if (!this.currentShard) {
      this.startNewShard();
    }

    // Check if this tensor should be transposed
    let writeData = data;
    let writeShape = metadata.shape;
    let layout: WeightLayout | undefined;
    let originalShape: number[] | undefined;

    if (this.transposeWeights && this.isMatmulWeight(name, metadata.shape)) {
      const bytesPerElement = this.getBytesPerElement(metadata.dtype);
      if (bytesPerElement > 0 && metadata.shape.length === 2) {
        const [rows, cols] = metadata.shape;
        writeData = this.transpose2D(data, rows, cols, metadata.dtype);
        writeShape = [cols, rows]; // Transposed shape
        layout = 'column';
        originalShape = metadata.shape;
      }
    }

    const alignedOffset = alignOffset(this.currentShardOffset);
    const spaceNeeded = (alignedOffset - this.currentShardOffset) + writeData.length;

    if (this.currentShardOffset > 0 && this.currentShardOffset + spaceNeeded > this.shardSize) {
      await this.finalizeShard();
      this.startNewShard();
    }

    const paddingNeeded = alignOffset(this.currentShardOffset) - this.currentShardOffset;
    if (paddingNeeded > 0) {
      this.currentShard!.data.push(createPadding(paddingNeeded));
      this.currentShard!.size += paddingNeeded;
      this.currentShardOffset += paddingNeeded;
    }

    const location: TensorLocation = {
      shardIndex: this.currentShardIndex,
      offset: this.currentShardOffset,
      size: writeData.length,
      shape: writeShape,
      dtype: metadata.dtype,
      layout,
      originalShape,
    };
    this.tensorLocations.set(name, location);

    let remaining = writeData;
    const tensorShards: Array<{ shardIndex: number; offset: number; size: number }> = [];

    while (remaining.length > 0) {
      const spaceInShard = this.shardSize - this.currentShardOffset;
      const writeSize = Math.min(remaining.length, spaceInShard);

      const chunk = remaining.slice(0, writeSize);
      this.currentShard!.data.push(chunk);
      this.currentShard!.size += writeSize;
      this.currentShardOffset += writeSize;

      tensorShards.push({
        shardIndex: this.currentShardIndex,
        offset: this.currentShardOffset - writeSize,
        size: writeSize,
      });

      remaining = remaining.slice(writeSize);

      if (remaining.length > 0) {
        await this.finalizeShard();
        this.startNewShard();
      }
    }

    if (tensorShards.length > 1) {
      location.spans = tensorShards;
    }

    // Track expert tensor for MoE models
    const shardIndices = tensorShards.map(s => s.shardIndex);
    this.trackExpertTensor(name, shardIndices, data.length);

    return location;
  }

  setConfig(config: Record<string, unknown>): void {
    this.manifest.config = config;
  }

  setTokenizer(tokenizer: Record<string, unknown>): void {
    this.manifest.tokenizer = tokenizer;
  }

  setMoEConfig(moeConfig: MoEConfig): void {
    this.manifest.moeConfig = moeConfig;
  }

  /**
   * Set conversion metadata - how this model was generated.
   * Enables reproducibility and debugging.
   */
  setConversion(conversion: ConversionInfo): void {
    this.manifest.conversion = conversion;
  }

  /**
   * Set runtime optimizations including kernel hints.
   * These are embedded in the manifest as defaults; YAML profiles can override at runtime.
   */
  setOptimizations(optimizations: RuntimeOptimizations): void {
    this.manifest.optimizations = optimizations;
  }

  setMetadata(meta: Record<string, unknown>): void {
    Object.assign(this.manifest, meta);
  }

  async writeTokenizer(tokenizer: TokenizerConfig): Promise<void> {
    if (!tokenizer.tokens || tokenizer.tokens.length === 0) {
      console.warn('[RDRRWriter] No vocab tokens found, skipping tokenizer bundling');
      this.manifest.tokenizer = {
        model: tokenizer.model,
        bosTokenId: tokenizer.bosTokenId,
        eosTokenId: tokenizer.eosTokenId,
        padTokenId: tokenizer.padTokenId,
        unkTokenId: tokenizer.unkTokenId,
        addBosToken: tokenizer.addBosToken,
        addEosToken: tokenizer.addEosToken,
      };
      return;
    }

    const vocab: Record<string, number> = {};
    for (let i = 0; i < tokenizer.tokens.length; i++) {
      vocab[tokenizer.tokens[i]] = i;
    }

    const hasMerges = tokenizer.merges && tokenizer.merges.length > 0;
    const hasScores = tokenizer.scores && tokenizer.scores.length > 0;
    const type = hasMerges ? 'bpe' : (hasScores ? 'unigram' : 'bpe');

    const tokenizerJson = {
      type,
      model: tokenizer.model,
      vocab,
      vocabSize: tokenizer.tokens.length,
      merges: hasMerges ? tokenizer.merges : null,
      scores: hasScores ? tokenizer.scores : null,
      tokenTypes: tokenizer.tokenTypes ?? null,
      specialTokens: {
        bos: tokenizer.bosTokenId,
        eos: tokenizer.eosTokenId,
        pad: tokenizer.padTokenId,
        unk: tokenizer.unkTokenId,
        sep: tokenizer.sepTokenId,
        cls: tokenizer.clsTokenId,
        mask: tokenizer.maskTokenId,
      },
      addBosToken: tokenizer.addBosToken ?? true,
      addEosToken: tokenizer.addEosToken ?? false,
      addSpacePrefix: tokenizer.addSpacePrefix ?? true,
    };

    const tokenizerPath = join(this.outputDir, 'tokenizer.json');
    await writeFile(tokenizerPath, JSON.stringify(tokenizerJson));

    console.log(`[RDRRWriter] Wrote tokenizer.json (${tokenizer.tokens.length} tokens, type: ${type})`);

    this.manifest.tokenizer = {
      type: 'bundled',
      file: 'tokenizer.json',
      vocabSize: tokenizer.tokens.length,
      tokenizerType: type,
    };
  }

  async writeHuggingFaceTokenizer(tokenizerJson: HuggingFaceTokenizer): Promise<void> {
    if (!tokenizerJson || !tokenizerJson.model) {
      console.warn('[RDRRWriter] Invalid HuggingFace tokenizer.json, skipping');
      return;
    }

    const tokenizerPath = join(this.outputDir, 'tokenizer.json');
    await writeFile(tokenizerPath, JSON.stringify(tokenizerJson));

    const model = tokenizerJson.model;
    let vocabSize: number;
    if (Array.isArray(model.vocab)) {
      vocabSize = model.vocab.length;
    } else {
      vocabSize = Object.keys(model.vocab || {}).length;
    }
    const type = model.type?.toLowerCase() || 'bpe';

    console.log(`[RDRRWriter] Wrote HuggingFace tokenizer.json (${vocabSize} tokens, type: ${type})`);

    this.manifest.tokenizer = {
      type: 'huggingface',
      file: 'tokenizer.json',
      vocabSize,
      tokenizerType: type,
    };
  }

  async finalize(): Promise<WriteResult> {
    // Warn about any unpaired gate/up tensors
    if (this.ffnFusionBuffer.size > 0) {
      for (const [layerKey, buffer] of this.ffnFusionBuffer) {
        if (buffer.gate && !buffer.up) {
          console.warn(`[RDRRWriter] Warning: Layer ${layerKey} has gate_proj but no up_proj - writing unfused`);
          await this.writeTensorInternal(buffer.gate.name, buffer.gate.data, buffer.gate.metadata);
        }
        if (buffer.up && !buffer.gate) {
          console.warn(`[RDRRWriter] Warning: Layer ${layerKey} has up_proj but no gate_proj - writing unfused`);
          await this.writeTensorInternal(buffer.up.name, buffer.up.data, buffer.up.metadata);
        }
      }
      this.ffnFusionBuffer.clear();
    }

    await this.finalizeShard();

    const tensors: Record<string, TensorLocation> = {};

    for (const [name, location] of this.tensorLocations) {
      tensors[name] = {
        shardIndex: location.shardIndex,
        offset: location.offset,
        size: location.size,
        shape: location.shape,
        dtype: location.dtype,
      };
      if (location.spans) {
        tensors[name].spans = location.spans;
      }
      if (location.layout) {
        tensors[name].layout = location.layout;
      }
      if (location.originalShape) {
        tensors[name].originalShape = location.originalShape;
      }
    }

    this.manifest.shards = this.shards.map(s => ({
      index: s.index,
      fileName: s.fileName,
      size: s.size,
      hash: s.hash,
      hashAlgorithm: s.hashAlgorithm,
    }));
    this.manifest.tensors = tensors;
    this.manifest.totalSize = this.shards.reduce((sum, s) => sum + s.size, 0);
    this.manifest.tensorCount = this.tensorLocations.size;

    // Populate expert shard mapping if MoE model
    if (this.manifest.moeConfig && this.expertShardMap.size > 0) {
      const expertShardMap: Record<string, number[]> = {};
      const expertTensors: Record<string, string[]> = {};

      for (const [key, shards] of this.expertShardMap) {
        expertShardMap[key] = Array.from(shards).sort((a, b) => a - b);
      }

      for (const [key, tensorNames] of this.expertTensorMap) {
        expertTensors[key] = tensorNames;
      }

      // Calculate average expert size for memory planning
      const expertSizes = Array.from(this.expertBytesMap.values());
      const expertBytes = expertSizes.length > 0
        ? Math.ceil(expertSizes.reduce((a, b) => a + b, 0) / expertSizes.length)
        : 0;

      this.manifest.moeConfig.expertShardMap = expertShardMap;
      this.manifest.moeConfig.expertTensors = expertTensors;
      this.manifest.moeConfig.expertBytes = expertBytes;

      // Include shared expert indices if any were detected
      if (this.sharedExpertIndices.size > 0) {
        this.manifest.moeConfig.sharedExperts = Array.from(this.sharedExpertIndices).sort((a, b) => a - b);
        console.log(`[RDRRWriter] Shared experts: ${this.manifest.moeConfig.sharedExperts.join(', ')}`);
      }

      console.log(`[RDRRWriter] MoE expert mapping: ${this.expertShardMap.size} experts, ~${(expertBytes / 1024 / 1024).toFixed(1)}MB each`);
    }

    const manifestPath = join(this.outputDir, 'manifest.json');
    await writeFile(manifestPath, JSON.stringify(this.manifest, null, 2));

    return {
      manifestPath,
      shardCount: this.shards.length,
      totalSize: this.manifest.totalSize,
      tensorCount: this.manifest.tensorCount,
    };
  }

  async cleanup(): Promise<void> {
    try {
      await rm(this.outputDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}

export interface WriteRDRROptions extends WriterOptions {
  onProgress?: (event: ProgressEvent) => void;
  /** Conversion metadata - how the model was generated */
  conversion?: ConversionInfo;
  /** Runtime optimizations including kernel hints */
  optimizations?: RuntimeOptimizations;
}

export async function writeRDRR(
  outputDir: string,
  modelInfo: ModelInfo,
  getTensorData: (tensor: TensorInfo) => Promise<ArrayBuffer>,
  options: WriteRDRROptions = {}
): Promise<WriteResult> {
  const config = modelInfo.config as Record<string, unknown> | undefined;
  const writer = new RDRRWriter(outputDir, {
    modelId: modelInfo.modelName || (config?.modelId as string) || 'model',
    architecture: modelInfo.architecture || (config?.architectures as string[])?.[0] || 'llama',
    quantization: modelInfo.quantization || options.quantization || 'Q4_K_M',
    ...options,
  });

  try {
    await writer.init();

    if (modelInfo.config) {
      writer.setConfig(modelInfo.config);
    }

    const tokenizer = modelInfo.tokenizer || modelInfo.tokenizerConfig;
    const hfTokenizer = modelInfo.tokenizerJson;

    if (hfTokenizer && hfTokenizer.model?.vocab) {
      await writer.writeHuggingFaceTokenizer(hfTokenizer);
    } else if (tokenizer?.tokens && tokenizer.tokens.length > 0) {
      await writer.writeTokenizer(tokenizer);
    } else if (tokenizer) {
      writer.setTokenizer(tokenizer as Record<string, unknown>);
    }

    if (config?.expertCount || config?.num_local_experts) {
      const numExperts = (config.expertCount || config.num_local_experts) as number;
      const numExpertsPerToken = (
        config.expertUsedCount ||
        config.num_experts_per_tok ||
        config.experts_per_token ||
        2
      ) as number;
      writer.setMoEConfig({ numExperts, numExpertsPerToken });
      console.log(`  MoE config: ${numExperts} experts, ${numExpertsPerToken} active per token`);
    }

    // Set conversion metadata if provided
    if (options.conversion) {
      writer.setConversion(options.conversion);
    }

    // Set runtime optimizations (including kernel hints) if provided
    if (options.optimizations) {
      writer.setOptimizations(options.optimizations);
    }

    const progressCallback = options.onProgress || (() => {});
    const totalTensors = modelInfo.tensors.length;

    for (let i = 0; i < modelInfo.tensors.length; i++) {
      const tensor = modelInfo.tensors[i];
      const data = await getTensorData(tensor);

      await writer.writeTensor(tensor.name, new Uint8Array(data), {
        shape: tensor.shape,
        dtype: tensor.dtype,
      });

      progressCallback({
        stage: 'writing',
        current: i + 1,
        total: totalTensors,
        tensorName: tensor.name,
      });
    }

    const result = await writer.finalize();
    progressCallback({ stage: 'complete', ...result });

    return result;
  } catch (error) {
    await writer.cleanup();
    throw error;
  }
}

export async function createTestModel(outputDir: string): Promise<WriteResult> {
  const writer = new RDRRWriter(outputDir, {
    modelId: 'tiny-test',
    architecture: 'test',
    quantization: 'F32',
  });

  await writer.init();

  writer.setConfig({
    vocabSize: 1000,
    hiddenSize: 64,
    numLayers: 2,
    numHeads: 2,
    contextLength: 128,
  });

  writer.setTokenizer({
    model: 'bpe',
    vocabSize: 1000,
    bosTokenId: 1,
    eosTokenId: 2,
  });

  const hiddenSize = 64;
  const vocabSize = 1000;
  const intermediateSize = 256;

  const embedData = new Float32Array(vocabSize * hiddenSize);
  for (let i = 0; i < embedData.length; i++) {
    embedData[i] = (Math.random() - 0.5) * 0.02;
  }
  await writer.writeTensor('embed_tokens.weight', new Uint8Array(embedData.buffer), {
    shape: [vocabSize, hiddenSize],
    dtype: 'F32',
  });

  for (let layer = 0; layer < 2; layer++) {
    const qkvSize = hiddenSize * hiddenSize * 3;
    const qkvData = new Float32Array(qkvSize);
    for (let i = 0; i < qkvSize; i++) {
      qkvData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.attention.qkv.weight`, new Uint8Array(qkvData.buffer), {
      shape: [hiddenSize * 3, hiddenSize],
      dtype: 'F32',
    });

    const oData = new Float32Array(hiddenSize * hiddenSize);
    for (let i = 0; i < oData.length; i++) {
      oData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.attention.o.weight`, new Uint8Array(oData.buffer), {
      shape: [hiddenSize, hiddenSize],
      dtype: 'F32',
    });

    const upData = new Float32Array(intermediateSize * hiddenSize);
    for (let i = 0; i < upData.length; i++) {
      upData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.ffn.up.weight`, new Uint8Array(upData.buffer), {
      shape: [intermediateSize, hiddenSize],
      dtype: 'F32',
    });

    const downData = new Float32Array(hiddenSize * intermediateSize);
    for (let i = 0; i < downData.length; i++) {
      downData[i] = (Math.random() - 0.5) * 0.02;
    }
    await writer.writeTensor(`layers.${layer}.ffn.down.weight`, new Uint8Array(downData.buffer), {
      shape: [hiddenSize, intermediateSize],
      dtype: 'F32',
    });

    const normData = new Float32Array(hiddenSize).fill(1.0);
    await writer.writeTensor(`layers.${layer}.input_norm.weight`, new Uint8Array(normData.buffer), {
      shape: [hiddenSize],
      dtype: 'F32',
    });
    await writer.writeTensor(`layers.${layer}.post_norm.weight`, new Uint8Array(normData.buffer), {
      shape: [hiddenSize],
      dtype: 'F32',
    });
  }

  const lmHeadData = new Float32Array(vocabSize * hiddenSize);
  for (let i = 0; i < lmHeadData.length; i++) {
    lmHeadData[i] = (Math.random() - 0.5) * 0.02;
  }
  await writer.writeTensor('lm_head.weight', new Uint8Array(lmHeadData.buffer), {
    shape: [vocabSize, hiddenSize],
    dtype: 'F32',
  });

  const finalNormData = new Float32Array(hiddenSize).fill(1.0);
  await writer.writeTensor('final_norm.weight', new Uint8Array(finalNormData.buffer), {
    shape: [hiddenSize],
    dtype: 'F32',
  });

  return writer.finalize();
}

export { DEFAULT_SHARD_SIZE, ALIGNMENT, computeHash };
