#!/usr/bin/env node
/**
 * Node.js Model Converter - Convert HuggingFace/GGUF models to RDRR format.
 *
 * Uses config-as-code pattern: model families detected via JSON presets.
 *
 * Usage:
 *   npx tsx converter/node-converter.ts <input> <output> [options]
 *
 * Examples:
 *   npx tsx converter/node-converter.ts ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/HASH/  models/gemma-1b
 *   npx tsx converter/node-converter.ts model.gguf models/my-model --quantize q4_k_m
 *   npx tsx converter/node-converter.ts --test ./test-model  # Create tiny test fixture
 *
 * @module converter/node-converter
 */

import { resolve, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { stat, readdir, readFile, open } from 'fs/promises';
import {
  parseSafetensors,
  detectModelFormat,
  loadModelConfig,
  loadTokenizerConfig,
  readTensorData,
  type SafetensorsTensor,
} from '../formats/safetensors/index.js';
import {
  parseGGUF,
  parseGGUFFile,
  getTensors,
} from '../formats/gguf/index.js';
import {
  writeRDRR,
  createTestModel,
  type WriteResult,
  type WriterOptions,
} from './writer.js';
import { quantizeToQ4KM, float32ToFloat16 } from './quantizer.js';
import { shouldQuantize as shouldQuantizeCore, sanitizeModelId } from './core.js';

// Import config-as-code preset detection
import {
  detectPreset,
  resolvePreset,
  type ModelType,
  type RawModelConfigSchema,
  type QuantizationInfoSchema,
} from '../config/index.js';
import { log as debugLog } from '../debug/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface ConvertOptions {
  input: string;
  output: string;
  quantize: 'q4_k_m' | 'f16' | 'f32' | null;
  quantizeEmbeddings: boolean;
  shardSize: number;
  modelId: string | null;
  textOnly: boolean;
  fast: boolean;
  verbose: boolean;
  test: boolean;
  help: boolean;
}

function parseArgs(argv: string[]): ConvertOptions {
  const opts: ConvertOptions = {
    input: '',
    output: '',
    quantize: null,
    quantizeEmbeddings: false,
    shardSize: 64,
    modelId: null,
    textOnly: false,
    fast: false,
    verbose: false,
    test: false,
    help: false,
  };

  let positionalIndex = 0;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case '--help':
      case '-h':
        opts.help = true;
        break;
      case '--test':
        opts.test = true;
        if (argv[i + 1] && !argv[i + 1].startsWith('-')) {
          opts.output = argv[++i];
        }
        break;
      case '--quantize':
      case '-q':
        const qVal = argv[++i]?.toLowerCase();
        if (qVal === 'q4_k_m' || qVal === 'q4' || qVal === 'q4k') {
          opts.quantize = 'q4_k_m';
        } else if (qVal === 'f16') {
          opts.quantize = 'f16';
        } else if (qVal === 'f32') {
          opts.quantize = 'f32';
        }
        break;
      case '--quantize-embeddings':
        opts.quantizeEmbeddings = true;
        break;
      case '--shard-size':
        opts.shardSize = parseInt(argv[++i] || '64', 10);
        break;
      case '--model-id':
        opts.modelId = argv[++i] || null;
        break;
      case '--text-only':
        opts.textOnly = true;
        break;
      case '--fast':
        opts.fast = true;
        break;
      case '--verbose':
      case '-v':
        opts.verbose = true;
        break;
      default:
        if (!arg.startsWith('-')) {
          if (positionalIndex === 0) {
            opts.input = arg;
          } else if (positionalIndex === 1) {
            opts.output = arg;
          }
          positionalIndex++;
        }
        break;
    }
  }

  return opts;
}

/**
 * Detect model type from architecture string and config.
 * Uses config-as-code preset detection for model family identification.
 *
 * @returns Object with presetId and modelType for grouping strategy
 */
function detectModelTypeFromPreset(
  arch: string,
  config: Record<string, unknown>
): { presetId: string; modelType: ModelType } {
  // Cast config to RawModelConfigSchema for preset detection
  const rawConfig = config as RawModelConfigSchema;

  // Use preset detection (config-as-code pattern)
  const presetId = detectPreset(rawConfig, arch);
  const preset = resolvePreset(presetId);

  // Get modelType from preset, with fallback logic for MoE/hybrid
  let modelType: ModelType = preset.modelType || 'transformer';

  // Override for MoE if config indicates experts (presets may not cover all cases)
  if (config.num_local_experts || config.expertCount || config.num_experts) {
    if (config.n_shared_experts) {
      modelType = 'deepseek';
    } else if (arch.toLowerCase().includes('jamba')) {
      modelType = 'jamba';
    } else if (modelType === 'transformer') {
      modelType = 'mixtral'; // Generic MoE
    }
  }

  return { presetId, modelType };
}

function normalizeQuantTag(value: string | null | undefined): string {
  if (!value) return 'f16';
  const lower = value.toLowerCase();
  if (lower === 'q4_k_m' || lower === 'q4k' || lower === 'q4') return 'q4_k_m';
  if (lower === 'q6_k' || lower === 'q6k') return 'q6_k';
  if (lower === 'q8_0' || lower === 'q8') return 'q8_0';
  if (lower === 'f16' || lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'bf16' || lower === 'bfloat16') return 'bf16';
  if (lower === 'f32' || lower === 'fp32' || lower === 'float32') return 'f32';
  return lower;
}

function resolveManifestQuantization(quantize: ConvertOptions['quantize'], fallback: string): string {
  if (quantize === 'q4_k_m') return 'Q4_K_M';
  if (quantize === 'f16') return 'F16';
  if (quantize === 'f32') return 'F32';
  return fallback;
}

function isEmbeddingTensorName(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.includes('embed') ||
    lower.includes('tok_embeddings') ||
    lower.includes('token_embd')
  );
}

function isLmHeadTensorName(name: string): boolean {
  const lower = name.toLowerCase();
  if (lower.includes('lm_head')) return true;
  if (lower.endsWith('output.weight')) return true;
  return lower.includes('output') && lower.includes('weight') && !lower.includes('attn');
}

function findTensorDtype(
  tensors: Array<{ name: string; dtype: string }>,
  matcher: (name: string) => boolean
): string | null {
  const match = tensors.find((t) => matcher(t.name));
  return match?.dtype ?? null;
}

function buildVariantTag(info: QuantizationInfoSchema): string {
  const weights = info.weights;
  const embeddings = info.embeddings ?? info.weights;
  const parts = [`w${weights}`, `emb${embeddings}`];
  const lmHead = info.lmHead ?? embeddings;
  if (lmHead !== embeddings) {
    parts.push(`head${lmHead}`);
  }
  return parts.join('-');
}

function buildQuantizationInfo(
  opts: ConvertOptions,
  originalDtype: string,
  embedDtype: string | null,
  lmHeadDtype: string | null
): QuantizationInfoSchema {
  const weights = normalizeQuantTag(opts.quantize ?? originalDtype);
  let embeddings = normalizeQuantTag(embedDtype ?? originalDtype);
  let lmHead = normalizeQuantTag(lmHeadDtype ?? embeddings);

  if (opts.quantize === 'q4_k_m') {
    if (opts.quantizeEmbeddings) {
      embeddings = weights;
      lmHead = weights;
    }
  } else if (opts.quantize === 'f16') {
    embeddings = 'f16';
    lmHead = 'f16';
  } else if (opts.quantize === 'f32') {
    embeddings = 'f32';
    lmHead = 'f32';
  }

  const info: QuantizationInfoSchema = {
    weights,
    embeddings,
    lmHead: lmHead !== embeddings ? lmHead : undefined,
  };
  info.variantTag = buildVariantTag(info);
  return info;
}

function resolveModelId(modelId: string | null, baseName: string, variantTag: string | undefined): string {
  if (modelId) return modelId;
  const base = sanitizeModelId(baseName);
  return variantTag ? `${base}-${variantTag}` : base;
}

function printHelp(): void {
  console.log(`
Node.js Model Converter - Convert HuggingFace/GGUF models to RDRR format.

Usage:
  npx tsx converter/node-converter.ts <input> <output> [options]
  npx tsx converter/node-converter.ts --test [output]

Arguments:
  <input>               HuggingFace directory or GGUF file path
  <output>              Output directory for RDRR model

Options:
  --quantize, -q <type> Quantize weights: q4_k_m, f16, f32 (default: preserve)
  --quantize-embeddings Also quantize embedding table
  --shard-size <mb>     Shard size in MB (default: 64)
  --model-id <id>       Override model ID in manifest
  --text-only           Extract only text model from multimodal
  --fast                Pre-load all shards into memory (faster, more RAM)
  --verbose, -v         Show detailed progress
  --test [path]         Create tiny test fixture at path (default: ./test-model)
  --help, -h            Show this help

Examples:
  # Convert HuggingFace model with Q4 quantization
  npx tsx converter/node-converter.ts \\
    ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/ \\
    models/gemma-1b \\
    --quantize q4_k_m

  # Convert GGUF file
  npx tsx converter/node-converter.ts model.gguf models/my-model

  # Multimodal to text-only
  npx tsx converter/node-converter.ts \\
    ~/models/gemma-3-4b-it \\
    models/gemma-4b-text \\
    --text-only --quantize q4_k_m

  # Create tiny test fixture
  npx tsx converter/node-converter.ts --test ./test-model
`);
}

async function detectInputFormat(inputPath: string): Promise<'gguf' | 'safetensors' | 'unknown'> {
  const stats = await stat(inputPath);

  if (stats.isFile()) {
    if (inputPath.endsWith('.gguf')) {
      return 'gguf';
    }
    if (inputPath.endsWith('.safetensors')) {
      return 'safetensors';
    }
  }

  if (stats.isDirectory()) {
    const files = await readdir(inputPath);
    if (files.some(f => f.endsWith('.safetensors') || f.includes('model.safetensors.index.json'))) {
      return 'safetensors';
    }
    if (files.some(f => f.endsWith('.gguf'))) {
      return 'gguf';
    }
  }

  return 'unknown';
}

interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
}

interface ModelInfo {
  modelName?: string;
  architecture?: string;
  quantization?: string;
  config?: Record<string, unknown>;
  tokenizer?: Record<string, unknown>;
  tokenizerConfig?: Record<string, unknown>;
  tokenizerJson?: Record<string, unknown>;
  tensors: TensorInfo[];
}

async function convertSafetensors(
  inputPath: string,
  outputPath: string,
  opts: ConvertOptions
): Promise<WriteResult> {
  const verboseLog = (msg: string) => opts.verbose && debugLog.verbose('Convert', msg);

  verboseLog(`Parsing safetensors from: ${inputPath}`);
  const parsed = await parseSafetensors(inputPath);
  verboseLog(`Found ${parsed.tensors.length} tensors`);

  // Load config and tokenizer
  const config = await loadModelConfig(inputPath).catch(() => ({}));
  const tokenizerConfig = await loadTokenizerConfig(inputPath).catch(() => undefined);

  // Detect architecture and model type using preset system
  const configRec = config as Record<string, unknown>;
  const arch = (configRec.architectures as string[] | undefined)?.[0] ||
    (configRec.model_type as string | undefined) ||
    'llama';
  const { presetId, modelType } = detectModelTypeFromPreset(arch, config);

  verboseLog(`Architecture: ${arch}`);
  verboseLog(`Detected preset: ${presetId}`);
  verboseLog(`Model type: ${modelType}`);
  verboseLog(`Config: ${JSON.stringify(config, null, 2).slice(0, 500)}...`);

  // Filter text-only if requested
  let tensors = parsed.tensors;
  if (opts.textOnly) {
    const textOnlyPatterns = ['language_model', 'model.', 'lm_head', 'embed_tokens'];
    const visionPatterns = ['vision_', 'vision_tower', 'multi_modal_projector', 'image_newline'];

    tensors = tensors.filter((t: SafetensorsTensor) => {
      const isVision = visionPatterns.some(p => t.name.includes(p));
      if (isVision) {
        verboseLog(`Skipping vision tensor: ${t.name}`);
        return false;
      }
      return true;
    });
    verboseLog(`Filtered to ${tensors.length} text-only tensors`);
  }

  // Detect the original dtype from the first weight tensor (most are the same dtype)
  const firstWeight = tensors.find((t: SafetensorsTensor) => t.name.includes('.weight'));
  const originalDtype = firstWeight?.dtype?.toUpperCase() || 'F32';

  // Filter out tensors without valid shape (e.g., metadata entries)
  const validTensors = tensors.filter((t: SafetensorsTensor) => {
    if (!t.shape || !Array.isArray(t.shape)) {
      verboseLog(`Skipping tensor without shape: ${t.name}`);
      return false;
    }
    return true;
  });

  // Create modelInfo with adjusted dtypes for quantized tensors
  const modelInfo: ModelInfo = {
    modelName: opts.modelId || basename(inputPath),
    architecture: arch,
    quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : originalDtype,
    config,
    tokenizerConfig,
    tokenizerJson: parsed.tokenizerJson,
    tensors: validTensors.map((t: SafetensorsTensor) => {
      // Determine the output dtype based on quantization settings
      let outputDtype = t.dtype;
      if (opts.quantize === 'q4_k_m') {
        const isEmbedding = t.name.includes('embed') || t.name.includes('lm_head');
        if (shouldQuantizeCore(t.name, t.shape) || (isEmbedding && opts.quantizeEmbeddings)) {
          outputDtype = 'Q4_K_M';
        }
      } else if (opts.quantize === 'f16' && t.dtype !== 'F16') {
        outputDtype = 'F16';
      }
      return {
        name: t.name,
        shape: t.shape,
        dtype: outputDtype,
        size: t.size,
      };
    }),
  };

  // Create tensor data getter
  const tensorMap = new Map(validTensors.map((t: SafetensorsTensor) => [t.name, t]));

  const getTensorData = async (info: TensorInfo): Promise<ArrayBuffer> => {
    const tensor = tensorMap.get(info.name);
    if (!tensor) {
      throw new Error(`Tensor not found: ${info.name}`);
    }

    // Read raw tensor data from safetensors file
    // Ensure tensor has filePath set for reading
    const tensorWithPath = { ...tensor, filePath: tensor.filePath || tensor.shardPath || inputPath };
    const data = await readTensorData(tensorWithPath);

    // Quantize if requested (uses shared shouldQuantize logic)
    if (opts.quantize === 'q4_k_m') {
      const isEmbedding = info.name.includes('embed') || info.name.includes('lm_head');

      // Use shared logic + embedding override
      if (shouldQuantizeCore(info.name, info.shape) || (isEmbedding && opts.quantizeEmbeddings)) {
        verboseLog(`Quantizing ${info.name} to Q4_K_M`);

        // Convert to F32 based on source dtype
        let f32: Float32Array;
        if (tensor.dtype === 'BF16') {
          const bf16Data = new Uint16Array(data);
          f32 = new Float32Array(bf16Data.length);
          for (let i = 0; i < bf16Data.length; i++) {
            const bits = bf16Data[i] << 16;
            const f32View = new Float32Array(1);
            new Uint32Array(f32View.buffer)[0] = bits;
            f32[i] = f32View[0];
          }
        } else {
          f32 = new Float32Array(data);
        }

        const q4 = quantizeToQ4KM(f32, info.shape);
        return q4.quantized.buffer as ArrayBuffer;
      }
    }

    if (opts.quantize === 'f16' && tensor.dtype !== 'F16') {
      verboseLog(`Converting ${info.name} from ${tensor.dtype} to F16`);

      // Handle different input dtypes
      let f32: Float32Array;
      if (tensor.dtype === 'BF16') {
        // Convert BF16 to F32 first
        const bf16Data = new Uint16Array(data);
        f32 = new Float32Array(bf16Data.length);
        for (let i = 0; i < bf16Data.length; i++) {
          // BF16 to F32: shift left by 16 bits (BF16 is just truncated F32)
          const bits = bf16Data[i] << 16;
          const f32View = new Float32Array(1);
          new Uint32Array(f32View.buffer)[0] = bits;
          f32[i] = f32View[0];
        }
      } else if (tensor.dtype === 'F32') {
        f32 = new Float32Array(data);
      } else {
        // For other dtypes, assume F32 interpretation (may need extension)
        verboseLog(`Warning: Unknown dtype ${tensor.dtype}, treating as F32`);
        f32 = new Float32Array(data);
      }

      // Convert F32 to F16
      const f16 = new Uint16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        f16[i] = float32ToFloat16(f32[i]);
      }
      return f16.buffer;
    }

    return data;
  };

  // Write RDRR
  verboseLog(`Writing RDRR to: ${outputPath}`);
  const writerOpts: WriterOptions = {
    shardSize: opts.shardSize * 1024 * 1024,
    modelId: opts.modelId || basename(inputPath),
    modelType,
    architecture: arch,
    quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : originalDtype,
  };

  return writeRDRR(outputPath, modelInfo, getTensorData, writerOpts);
}

async function convertGGUF(
  inputPath: string,
  outputPath: string,
  opts: ConvertOptions
): Promise<WriteResult> {
  const verboseLog = (msg: string) => opts.verbose && debugLog.verbose('Convert', msg);

  verboseLog(`Parsing GGUF from: ${inputPath}`);
  const parsed = await parseGGUFFile(inputPath);
  verboseLog(`Found ${parsed.tensors.length} tensors`);

  // Use the already parsed GGUF config
  const ggufConfig = parsed.config;
  const arch = ggufConfig.architecture || parsed.architecture || 'llama';
  const modelName = parsed.modelName || basename(inputPath, '.gguf');

  // Map GGUF config to HuggingFace-style config that DOPPLER expects
  // (Build this first so we can detect model type from it)
  const config: Record<string, unknown> = {
    architectures: [arch + 'ForCausalLM'], // e.g., "gemma2" -> "gemma2ForCausalLM"
    model_type: arch,
    hidden_size: ggufConfig.embeddingLength,
    num_hidden_layers: ggufConfig.blockCount,
    num_attention_heads: ggufConfig.attentionHeadCount,
    num_key_value_heads: ggufConfig.attentionHeadCountKV,
    vocab_size: ggufConfig.vocabSize,
    intermediate_size: ggufConfig.feedForwardLength,
    rms_norm_eps: ggufConfig.attentionLayerNormRMSEpsilon || ggufConfig.attentionLayerNormEpsilon || 1e-5,
    rope_theta: ggufConfig.ropeFreqBase || 10000,
    max_position_embeddings: ggufConfig.contextLength || 8192,
    // MoE fields from GGUF
    num_local_experts: ggufConfig.expertCount,
  };

  // Detect model type using preset system
  const { presetId, modelType } = detectModelTypeFromPreset(arch, config);

  verboseLog(`Architecture: ${arch}`);
  verboseLog(`Detected preset: ${presetId}`);
  verboseLog(`Model type: ${modelType}`);
  verboseLog(`Model: ${modelName}`);

  const tensors = parsed.tensors;

  // Extract tokenizer from GGUF
  const ggufTokenizer = ggufConfig.tokenizer;
  verboseLog(`Tokenizer: ${ggufTokenizer?.model || 'unknown'}, ${ggufTokenizer?.tokens?.length || 0} tokens`);

  const modelInfo: ModelInfo = {
    modelName: opts.modelId || modelName,
    architecture: arch,
    quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : parsed.quantization || 'F32',
    config,
    // Pass GGUF tokenizer data for bundling
    tokenizer: ggufTokenizer ? {
      model: ggufTokenizer.model,
      tokens: ggufTokenizer.tokens,
      scores: ggufTokenizer.scores,
      tokenTypes: ggufTokenizer.tokenTypes,
      merges: ggufTokenizer.merges,
      bosTokenId: ggufTokenizer.bosTokenId,
      eosTokenId: ggufTokenizer.eosTokenId,
      padTokenId: ggufTokenizer.padTokenId,
      unkTokenId: ggufTokenizer.unkTokenId,
      sepTokenId: ggufTokenizer.sepTokenId,
      clsTokenId: ggufTokenizer.clsTokenId,
      maskTokenId: ggufTokenizer.maskTokenId,
      addBosToken: ggufTokenizer.addBosToken,
      addEosToken: ggufTokenizer.addEosToken,
      addSpacePrefix: ggufTokenizer.addSpacePrefix,
    } : undefined,
    tensors: tensors.map(t => ({
      name: t.name,
      shape: t.shape,
      dtype: t.dtype,
      size: t.size,
    })),
  };

  // Read tensor data
  const fileHandle = await open(inputPath, 'r');
  const getTensorData = async (info: TensorInfo): Promise<ArrayBuffer> => {
    const tensor = tensors.find(t => t.name === info.name);
    if (!tensor) {
      throw new Error(`Tensor not found: ${info.name}`);
    }

    const buffer = Buffer.alloc(tensor.size);
    await fileHandle.read(buffer, 0, tensor.size, tensor.offset);
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  };

  try {
    verboseLog(`Writing RDRR to: ${outputPath}`);
    const writerOpts: WriterOptions = {
      shardSize: opts.shardSize * 1024 * 1024,
      modelId: opts.modelId || modelName,
      modelType,
      architecture: arch,
      quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : parsed.quantization || 'F32',
    };

    const result = await writeRDRR(outputPath, modelInfo, getTensorData, writerOpts);
    return result;
  } finally {
    await fileHandle.close();
  }
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  // Handle --test mode
  if (opts.test) {
    const outputPath = opts.output || resolve(process.cwd(), 'test-model');
    console.log(`Creating test model fixture at: ${outputPath}`);
    const result = await createTestModel(outputPath);
    console.log(`\nTest fixture created:`);
    console.log(`  Manifest: ${result.manifestPath}`);
    console.log(`  Shards: ${result.shardCount}`);
    console.log(`  Tensors: ${result.tensorCount}`);
    console.log(`  Size: ${(result.totalSize / 1024).toFixed(1)} KB`);
    process.exit(0);
  }

  if (!opts.input || !opts.output) {
    console.error('Error: <input> and <output> are required');
    printHelp();
    process.exit(1);
  }

  const inputPath = resolve(opts.input);
  const outputPath = resolve(opts.output);

  // Validate input exists
  try {
    await stat(inputPath);
  } catch {
    console.error(`Error: Input path does not exist: ${inputPath}`);
    process.exit(1);
  }

  // Detect format
  const format = await detectInputFormat(inputPath);
  if (format === 'unknown') {
    console.error(`Error: Could not detect model format for: ${inputPath}`);
    console.error('Expected: HuggingFace directory or .gguf file');
    process.exit(1);
  }

  console.log(`Converting ${format.toUpperCase()} model...`);
  console.log(`  Input: ${inputPath}`);
  console.log(`  Output: ${outputPath}`);
  if (opts.quantize) {
    console.log(`  Quantization: ${opts.quantize}`);
  }

  try {
    let result: WriteResult;

    if (format === 'gguf') {
      result = await convertGGUF(inputPath, outputPath, opts);
    } else {
      result = await convertSafetensors(inputPath, outputPath, opts);
    }

    console.log(`\nConversion complete:`);
    console.log(`  Manifest: ${result.manifestPath}`);
    console.log(`  Shards: ${result.shardCount}`);
    console.log(`  Tensors: ${result.tensorCount}`);
    console.log(`  Size: ${(result.totalSize / (1024 * 1024)).toFixed(1)} MB`);
  } catch (error) {
    console.error(`\nConversion failed: ${(error as Error).message}`);
    if (opts.verbose) {
      console.error((error as Error).stack);
    }
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
