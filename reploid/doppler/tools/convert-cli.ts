#!/usr/bin/env node
/**
 * Model Conversion CLI - Converts GGUF or SafeTensors to RDRR format.
 */

import { readFile, stat, rm } from 'fs/promises';
import { basename, extname, resolve, dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

import { parseGGUFFile, type GGUFParseResult, type GGUFTensor, type GGUFConfig } from './gguf-parser.js';
import { parseSafetensors, readTensorData, type ParsedSafetensorsFile, type ParsedSafetensorsIndex, type SafetensorsTensor } from './safetensors-parser.js';
import { quantizeToQ4KM, quantizeToQ4KMRowWise, quantizeToQ4KMColumnWise, shouldQuantize, float16ToFloat32 } from './quantizer.js';
import { writeRDRR, createTestModel, type ProgressEvent, type TensorInfo as RDRRTensorInfo } from './rdrr-writer.js';
import type { ConversionInfo, RuntimeOptimizations, Q4KLayout, ComputePrecision } from '../storage/rdrr-format.js';

// Converter version for reproducibility
const CONVERTER_VERSION = '1.0.0';

export type InputFormat = 'gguf' | 'safetensors' | 'safetensors-dir' | 'safetensors-index';
export type QuantizationType = 'q4_k_m' | 'f16' | 'f32';

export interface ConvertOptions {
  input: string | null;
  output: string | null;
  quantize: QuantizationType | null;
  q4kLayout: Q4KLayout;  // Q4K block layout: 'flat', 'row_wise', or 'column_wise'
  computePrecision: ComputePrecision;  // Preferred compute precision: 'f16', 'f32', or 'auto'
  shardSize: number;
  modelId: string | null;
  test: boolean;
  verbose: boolean;
  fast: boolean;
  textOnly: boolean;
  quantizeEmbeddings: boolean;
  fuseGateUp: boolean;
  cleanOutput: boolean;  // Delete existing output directory before converting
  help: boolean;
}

export interface ConvertResult {
  shardCount: number;
  tensorCount: number;
  totalSize: number;
  manifestPath: string;
}

export interface ModelInfo {
  architecture?: string;
  modelName?: string;
  quantization?: string;
  tensors: TensorInfo[];
  config?: ModelConfig;
  fileSize?: number;
  shards?: Array<{ file: string; path: string; size: number }>;
}

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  offset: number;
  size: number;
  quantization?: string;
  filePath?: string;
  shardPath?: string;
}

export interface ModelConfig {
  architectures?: string[];
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  intermediate_size?: number;
  rope_theta?: number;
  text_config?: ModelConfig;
  vision_config?: Record<string, unknown>;
  quantization_config?: {
    modules_to_not_convert?: string[];
  };
  [key: string]: unknown;
}

function parseArgs(args: string[]): ConvertOptions {
  const options: ConvertOptions = {
    input: null,
    output: null,
    quantize: null,
    q4kLayout: 'column_wise',  // Default: column-wise (14% faster than flat, 2.7x faster than row)
    computePrecision: 'auto',  // Default: auto-detect GPU capabilities at runtime
    shardSize: 64,
    modelId: null,
    test: false,
    verbose: false,
    fast: false,
    textOnly: false,
    quantizeEmbeddings: false,
    fuseGateUp: false,
    cleanOutput: true,  // Default: clean output directory to avoid stale files
    help: false,
  };

  let i = 0;
  while (i < args.length) {
    const arg = args[i];

    if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else if (arg === '--test') {
      options.test = true;
    } else if (arg === '--verbose' || arg === '-v') {
      options.verbose = true;
    } else if (arg === '--fast') {
      options.fast = true;
    } else if (arg === '--text-only') {
      options.textOnly = true;
    } else if (arg === '--quantize-embeddings') {
      options.quantizeEmbeddings = true;
    } else if (arg === '--fuse-gate-up') {
      options.fuseGateUp = true;
    } else if (arg === '--no-clean') {
      options.cleanOutput = false;
    } else if (arg === '--clean') {
      options.cleanOutput = true;
    } else if (arg === '--quantize' || arg === '-q') {
      options.quantize = args[++i]?.toLowerCase() as QuantizationType;
    } else if (arg === '--q4k-layout') {
      const layout = args[++i]?.toLowerCase();
      if (layout === 'flat' || layout === 'row_wise' || layout === 'column_wise') {
        options.q4kLayout = layout;
      } else {
        console.warn(`Warning: Unknown Q4K layout '${layout}', using 'row_wise'`);
      }
    } else if (arg === '--compute-precision' || arg === '-cp') {
      const precision = args[++i]?.toLowerCase();
      if (precision === 'f16' || precision === 'f32' || precision === 'auto') {
        options.computePrecision = precision;
      } else {
        console.warn(`Warning: Unknown compute precision '${precision}', using 'auto'`);
      }
    } else if (arg === '--shard-size') {
      options.shardSize = parseInt(args[++i], 10);
    } else if (arg === '--model-id') {
      options.modelId = args[++i];
    } else if (!arg.startsWith('-')) {
      if (!options.input) {
        options.input = arg;
      } else if (!options.output) {
        options.output = arg;
      }
    }

    i++;
  }

  return options;
}

function printHelp(): void {
  console.log(`
DOPPLER Model Converter - Convert models to .rdrr format

Usage:
  node convert-cli.js <input> [output-dir] [options]

Arguments:
  input       Path to GGUF file, safetensors file, or HuggingFace model directory
  output-dir  Directory to write .rdrr output (default: ../models/<model-name>)

Options:
  --quantize <type>       Quantize weights (q4_k_m, f16, f32)
  --q4k-layout <layout>   Q4K block layout: column_wise (default), row_wise, or flat
                          column_wise: Blocks by input column (FASTEST - 8 tok/s, optimal for GEMV)
                          flat: Sequential packing (7.2 tok/s, simpler)
                          row_wise: Blocks by output row (3 tok/s, poor thread utilization)
  --compute-precision, -cp <prec>  Preferred compute precision: f16, f32, auto (default: auto)
                          f16: Fast F16 arithmetic (requires shader-f16, like WebLLM q4f16)
                          f32: Compatible F32 arithmetic (slower, works everywhere, like q4f32)
                          auto: Detect GPU capabilities at runtime
  --quantize-embeddings   Also quantize embedding table (saves ~50% for large vocabs)
  --fuse-gate-up          Fuse gate+up projections for 2-pass FFN (faster inference)
  --shard-size <mb>       Shard size in MB (default: 64)
  --model-id <id>         Override model ID in manifest (use standard naming below)
  --text-only             Extract only text model (skip vision/projector, strip prefixes)
  --no-clean              Don't delete existing output directory (default: clean before convert)
  --fast                  Pre-load shards into memory (faster, uses more RAM)
  --test                  Create tiny test model (ignores input)
  --verbose, -v           Verbose output
  --help, -h              Show this help

Model Naming Convention:
  {family}-{version}-{size}-{variant}-{quant}[-{compute}]

  Where:
    {quant}   = q4 (Q4_K_M), f16, f32
    {compute} = f16, f32, or omit for auto (optional, WebLLM-style suffix)

  Examples:
    gemma-3-1b-it-q4       Gemma 3, 1B, Q4_K_M, auto compute precision
    gemma-3-1b-it-q4-f16   Gemma 3, 1B, Q4_K_M, prefer F16 compute (like q4f16)
    gemma-3-1b-it-q4-f32   Gemma 3, 1B, Q4_K_M, prefer F32 compute (like q4f32)
    llama-3.2-1b-q4-f16    Llama 3.2, 1B, Q4_K_M, prefer F16 compute
    mistral-7b-v0.3-q4    Mistral 7B v0.3, Q4_K_M quantized

Examples:
  # Convert GGUF model
  node convert-cli.js tinyllama-1.1b-q4_k_m.gguf ./tinyllama-rdrr

  # Convert HuggingFace model with quantization
  node convert-cli.js ./Llama-3.2-1B ./llama-rdrr --quantize q4_k_m

  # Convert multimodal model to text-only
  node convert-cli.js ./gemma-3-4b-it ./gemma-4b-text --text-only --quantize q4_k_m

  # Create test fixture
  node convert-cli.js --test ./test-model

Supported formats:
  - GGUF (.gguf) - llama.cpp format
  - Safetensors (.safetensors) - HuggingFace format
  - HuggingFace directory (with model.safetensors or index.json)
`);
}

function progressBar(current: number, total: number, width = 40): string {
  const percent = current / total;
  const filled = Math.round(width * percent);
  const empty = width - filled;
  const bar = '█'.repeat(filled) + '░'.repeat(empty);
  return `[${bar}] ${Math.round(percent * 100)}%`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/**
 * Build conversion metadata for reproducibility.
 */
function buildConversionInfo(
  source: string,
  options: ConvertOptions,
  originalDtype?: string,
  quantizationOverride?: string,
  q4kLayoutOverride?: Q4KLayout | null
): ConversionInfo {
  // Reconstruct the CLI command for reproducibility
  const cmdParts = ['npx tsx doppler/tools/convert-cli.ts', source, options.output || '.'];
  if (options.quantize) cmdParts.push(`--quantize ${options.quantize}`);
  if (options.q4kLayout !== 'row_wise') cmdParts.push(`--q4k-layout ${options.q4kLayout}`);
  if (options.computePrecision !== 'auto') cmdParts.push(`--compute-precision ${options.computePrecision}`);
  if (options.fuseGateUp) cmdParts.push('--fuse-gate-up');
  if (options.modelId) cmdParts.push(`--model-id ${options.modelId}`);
  if (options.textOnly) cmdParts.push('--text-only');
  if (options.quantizeEmbeddings) cmdParts.push('--quantize-embeddings');

  const quantType = options.quantize
    ? options.quantize.toUpperCase()
    : (quantizationOverride || 'F16');
  const quantLower = quantType.toLowerCase();
  const q4kLayout = q4kLayoutOverride ?? (options.quantize === 'q4_k_m' ? options.q4kLayout : undefined);

  return {
    source,
    convertedAt: new Date().toISOString(),
    converterVersion: CONVERTER_VERSION,
    command: cmdParts.join(' '),
    quantization: {
      type: quantType,
      layout: quantLower.includes('q4') ? q4kLayout : undefined,
      fuseGateUp: options.fuseGateUp,
      quantizeEmbeddings: options.quantizeEmbeddings,
    },
    originalDtype,
  };
}

/**
 * Build default runtime optimizations based on model quantization and compute precision.
 * These are embedded in the manifest; YAML profiles can override at runtime.
 *
 * Based on benchmark findings:
 * - Q4K row-wise: dequant_f16 is 2x faster than fused_q4k (poor thread utilization in fused kernel)
 * - F16 weights: gemv_subgroup with multicol for large N
 *
 * Compute precision follows WebLLM convention:
 * - f16: Like q4f16 - fast F16 arithmetic (requires shader-f16)
 * - f32: Like q4f32 - compatible F32 arithmetic
 * - auto: Detect at runtime
 */
function buildOptimizations(options: ConvertOptions, quantizationOverride?: string): RuntimeOptimizations {
  const quantLabel = options.quantize ? options.quantize.toLowerCase() : (quantizationOverride || '').toLowerCase();
  const isQ4K = quantLabel.includes('q4_k') || quantLabel.includes('q4k');
  const precision = options.computePrecision;

  // Determine Q4K matmul strategy based on compute precision
  let q4kMatmul: 'dequant_f16' | 'dequant_f32' | undefined;
  if (isQ4K) {
    if (precision === 'f32') {
      q4kMatmul = 'dequant_f32';  // Force F32 compute (like q4f32)
    } else {
      q4kMatmul = 'dequant_f16';  // Default to F16 dequant (faster, like q4f16)
    }
  }

  return {
    kernelHints: {
      // Global compute precision preference
      computePrecision: precision,
      // Q4K: use dequant path with specified precision
      q4kMatmul,
      // F16: use subgroup GEMV
      f16Matmul: 'gemv_subgroup',
      // Default attention kernels
      attentionPrefill: 'tiled_large',
      attentionDecode: 'streaming',
    },
  };
}

async function detectFormat(inputPath: string): Promise<InputFormat> {
  const stats = await stat(inputPath);

  if (stats.isDirectory()) {
    return 'safetensors-dir';
  }

  const ext = extname(inputPath).toLowerCase();
  if (ext === '.gguf') {
    return 'gguf';
  }
  if (ext === '.safetensors') {
    return 'safetensors';
  }
  if (ext === '.json' && inputPath.includes('index')) {
    return 'safetensors-index';
  }

  throw new Error(`Unknown input format: ${inputPath}`);
}

function inferConfigFromTensors(modelInfo: ModelInfo): void {
  if (!modelInfo.config) {
    modelInfo.config = {};
  }
  const config = modelInfo.config;

  const qProj = modelInfo.tensors.find(t => t.name.includes('.0.self_attn.q_proj.weight'));
  const kProj = modelInfo.tensors.find(t => t.name.includes('.0.self_attn.k_proj.weight'));
  const qNorm = modelInfo.tensors.find(t => t.name.includes('.0.self_attn.q_norm.weight'));

  const embed = modelInfo.tensors.find(t => t.name.includes('embed_tokens.weight'));
  if (embed && !config.hidden_size) {
    config.hidden_size = embed.shape[1];
    console.log(`  Inferred hidden_size=${config.hidden_size} from embedding`);
  }

  if (!config.num_hidden_layers) {
    const layerNums = modelInfo.tensors
      .filter(t => t.name.includes('.layers.'))
      .map(t => {
        const match = t.name.match(/\.layers\.(\d+)\./);
        return match ? parseInt(match[1]) : -1;
      })
      .filter(n => n >= 0);
    if (layerNums.length > 0) {
      config.num_hidden_layers = Math.max(...layerNums) + 1;
      console.log(`  Inferred num_hidden_layers=${config.num_hidden_layers}`);
    }
  }

  if (qNorm && !config.head_dim) {
    config.head_dim = qNorm.shape[0];
    console.log(`  Inferred head_dim=${config.head_dim} from q_norm`);
  }

  if (qProj && config.head_dim && !config.num_attention_heads) {
    config.num_attention_heads = qProj.shape[0] / config.head_dim;
    console.log(`  Inferred num_attention_heads=${config.num_attention_heads} from q_proj`);
  }
  if (kProj && config.head_dim && !config.num_key_value_heads) {
    config.num_key_value_heads = kProj.shape[0] / config.head_dim;
    console.log(`  Inferred num_key_value_heads=${config.num_key_value_heads} from k_proj`);
  }

  const gateProj = modelInfo.tensors.find(t => t.name.includes('.0.mlp.gate_proj.weight'));
  if (gateProj && !config.intermediate_size) {
    config.intermediate_size = gateProj.shape[0];
    console.log(`  Inferred intermediate_size=${config.intermediate_size} from gate_proj`);
  }

  // Detect architecture for defaults
  const arch = config.architectures?.[0] || '';
  const isGemma = /gemma/i.test(arch);
  const isQwen3 = /qwen.*3|qwen3/i.test(arch);
  const isKimiK2 = /kimi.*k2|kimi_k2/i.test(arch);
  const isMixtral = /mixtral/i.test(arch);
  const isGptOss = /gptoss|gpt.*oss/i.test(arch);

  // RoPE theta defaults by architecture
  let expectedTheta = 10000;
  if (isGemma || isQwen3 || isMixtral) {
    expectedTheta = 1000000;
  } else if (isKimiK2) {
    expectedTheta = 50000;  // Kimi K2 uses YARN 32x scaling
  } else if (isGptOss) {
    expectedTheta = 150000;
  }

  if (!config.rope_theta) {
    config.rope_theta = expectedTheta;
    console.log(`  Set rope_theta=${config.rope_theta} for ${arch || 'unknown'}`);
  } else if (config.rope_theta !== expectedTheta &&
             (arch.includes('Mistral') || arch.includes('Llama'))) {
    console.warn(`  Warning: rope_theta=${config.rope_theta} unusual for ${arch}, expected ${expectedTheta}`);
  }

  if (!config.head_dim && config.hidden_size && config.num_attention_heads) {
    if (arch.includes('Gemma3') || arch.includes('Gemma 3')) {
      config.head_dim = 256;
    } else {
      config.head_dim = Math.floor(config.hidden_size / config.num_attention_heads);
    }
    console.log(`  Inferred head_dim=${config.head_dim}`);
  }

  // ==========================================================================
  // DOPPLER-specific config (architectural quirks for runtime)
  // These are set explicitly in manifest so pipeline doesn't need name-matching
  // ==========================================================================

  // Gemma family: scale embeddings by sqrt(hidden_size), GELU activation, high RoPE theta
  if (isGemma) {
    config.scale_embeddings = true;
    config.rms_norm_weight_offset = true;  // Gemma 3 adds +1 to norm weights
    config.hidden_activation = config.hidden_activation ?? 'gelu';  // Gemma uses GELU, not SiLU
    console.log(`  Set Gemma defaults: scale_embeddings=true, rms_norm_weight_offset=true, hidden_activation=gelu`);
  }

  // Qwen3 MoE: uses SiLU activation, standard norm weights
  if (isQwen3) {
    config.scale_embeddings = false;
    config.rms_norm_weight_offset = false;
    config.hidden_activation = 'silu';
    console.log(`  Set Qwen3 defaults: scale_embeddings=false, hidden_activation=silu`);
  }

  // Kimi K2: uses MLA (Multi-head Latent Attention), SwiGLU, YARN scaling
  // 1T params, 32B active, 384 experts + 1 shared, top-8
  if (isKimiK2) {
    config.scale_embeddings = false;
    config.rms_norm_weight_offset = false;
    config.hidden_activation = 'silu';
    console.log(`  Set Kimi K2 defaults: scale_embeddings=false, hidden_activation=silu (MLA attention)`);
  }

  // Mixtral: standard MoE with SiLU, high rope_theta
  if (isMixtral) {
    config.scale_embeddings = false;
    config.rms_norm_weight_offset = false;
    config.hidden_activation = 'silu';
    console.log(`  Set Mixtral defaults: scale_embeddings=false, hidden_activation=silu`);
  }

  // GPT-OSS: standard LLaMA-like architecture
  if (isGptOss) {
    config.scale_embeddings = false;
    config.rms_norm_weight_offset = false;
    config.hidden_activation = 'silu';
    console.log(`  Set GPT-OSS defaults: scale_embeddings=false, hidden_activation=silu`);
  }

  const required = ['hidden_size', 'num_hidden_layers', 'num_attention_heads'];
  const missing = required.filter(k => !config[k]);
  if (missing.length > 0) {
    console.warn(`  Warning: Could not infer config values: ${missing.join(', ')}`);
  }
}

function bf16ToF32(bf16: number): number {
  const f32View = new Float32Array(1);
  const u32View = new Uint32Array(f32View.buffer);
  u32View[0] = bf16 << 16;
  return f32View[0];
}

function convertToF32(buffer: ArrayBuffer, dtype: string): Float32Array {
  if (dtype === 'F32') {
    return new Float32Array(buffer);
  }

  if (dtype === 'F16') {
    const f16 = new Uint16Array(buffer);
    const f32 = new Float32Array(f16.length);
    for (let i = 0; i < f16.length; i++) {
      f32[i] = float16ToFloat32(f16[i]);
    }
    return f32;
  }

  if (dtype === 'BF16') {
    const bf16 = new Uint16Array(buffer);
    const f32 = new Float32Array(bf16.length);
    for (let i = 0; i < bf16.length; i++) {
      f32[i] = bf16ToF32(bf16[i]);
    }
    return f32;
  }

  throw new Error(`Unsupported dtype for conversion: ${dtype}`);
}

async function convertGGUF(inputPath: string, outputDir: string, options: ConvertOptions): Promise<ConvertResult> {
  console.log(`\nParsing GGUF: ${inputPath}`);

  const parsed = await parseGGUFFile(inputPath);
  console.log(`  Architecture: ${parsed.architecture}`);
  console.log(`  Quantization: ${parsed.quantization}`);
  console.log(`  Tensors: ${parsed.tensors.length}`);
  console.log(`  File size: ${formatBytes(parsed.fileSize)}`);

  const fileBuffer = await readFile(inputPath);

  const modelInfo: ModelInfo = {
    architecture: parsed.architecture,
    modelName: parsed.modelName,
    quantization: parsed.quantization,
    tensors: parsed.tensors.map(t => ({
      name: t.name,
      shape: t.shape,
      dtype: t.dtype,
      offset: t.offset,
      size: t.size,
      quantization: parsed.quantization,
    })),
    config: parsed.config as unknown as ModelConfig,
    fileSize: parsed.fileSize,
  };

  const getTensorData = async (tensor: RDRRTensorInfo): Promise<ArrayBuffer> => {
    // Find full tensor info from our local array
    const localTensor = modelInfo.tensors.find(t => t.name === tensor.name)!;
    const data = fileBuffer.buffer.slice(localTensor.offset, localTensor.offset + localTensor.size) as ArrayBuffer;
    const modulesToNotConvert = modelInfo.config?.quantization_config?.modules_to_not_convert ?? null;

    if (options.quantize === 'q4_k_m' && shouldQuantize(tensor.name, tensor.shape, {
      quantizeEmbeddings: options.quantizeEmbeddings,
      modulesToNotConvert,
    })) {
      const sourceQuant = localTensor.quantization || modelInfo.quantization;

      if (sourceQuant === 'F16' || sourceQuant === 'F32' || sourceQuant === 'BF16') {
        const f32Data = convertToF32(data, sourceQuant);

        // Use appropriate Q4K layout for 2D weight matrices
        let quantized: Uint8Array;

        if (tensor.shape.length === 2 && options.q4kLayout !== 'flat') {
          const shape2D = tensor.shape as [number, number];

          if (options.q4kLayout === 'column_wise') {
            const result = quantizeToQ4KMColumnWise(f32Data, shape2D);
            quantized = result.quantized;
            if (options.verbose) {
              console.log(`  Re-quantized ${tensor.name}: ${sourceQuant} -> Q4_K_M (column-wise, transposed)`);
            }
          } else {
            const result = quantizeToQ4KMRowWise(f32Data, shape2D);
            quantized = result.quantized;
            if (options.verbose) {
              console.log(`  Re-quantized ${tensor.name}: ${sourceQuant} -> Q4_K_M (row-wise)`);
            }
          }
        } else {
          const result = quantizeToQ4KM(f32Data, tensor.shape);
          quantized = result.quantized;
          if (options.verbose) {
            console.log(`  Re-quantized ${tensor.name}: ${sourceQuant} -> Q4_K_M`);
          }
        }

        localTensor.dtype = 'Q4_K_M';
        localTensor.size = quantized.length;
        return quantized.buffer as ArrayBuffer;
      }
    }

    return data;
  };

  inferConfigFromTensors(modelInfo);

  const sourceQuant = (parsed.quantization || '').toUpperCase();
  const sourceIsFloat = sourceQuant === 'F16' || sourceQuant === 'F32' || sourceQuant === 'BF16';
  const outputQuant = options.quantize ? options.quantize.toUpperCase() : (sourceQuant || 'UNKNOWN');
  const outputQuantLower = outputQuant.toLowerCase();
  const outputIsQ4K = outputQuantLower.includes('q4_k') || outputQuantLower.includes('q4k');
  let resolvedQ4KLayout: Q4KLayout | null = null;

  if (outputIsQ4K) {
    if (options.quantize === 'q4_k_m' && sourceIsFloat) {
      resolvedQ4KLayout = options.q4kLayout;
    } else if (sourceQuant.toLowerCase().includes('q4_k')) {
      resolvedQ4KLayout = 'row_wise';
      if (options.quantize === 'q4_k_m' && options.q4kLayout !== 'row_wise' && !sourceIsFloat) {
        console.warn(`  Q4K layout ignored (GGUF already quantized): forcing row_wise`);
      }
    } else {
      resolvedQ4KLayout = options.q4kLayout;
    }
  }

  // Add Q4K layout to config if quantized
  if (resolvedQ4KLayout && resolvedQ4KLayout !== 'flat') {
    modelInfo.config = modelInfo.config || {};
    (modelInfo.config as Record<string, unknown>).q4kLayout = resolvedQ4KLayout;
    console.log(`  Q4K layout: ${resolvedQ4KLayout}`);
  }

  // Build conversion metadata and runtime optimizations
  const conversion = buildConversionInfo(inputPath, options, 'gguf', outputQuant, resolvedQ4KLayout);
  const optimizations = buildOptimizations(options, outputQuant);

  const result = await writeRDRR(outputDir, modelInfo, getTensorData, {
    modelId: options.modelId || modelInfo.modelName || 'unknown',
    quantization: outputQuant || modelInfo.quantization || 'unknown',
    shardSize: options.shardSize * 1024 * 1024,
    fuseGateUp: options.fuseGateUp,
    conversion,
    optimizations,
    onProgress: (progress: ProgressEvent) => {
      if (progress.stage === 'writing' && options.verbose) {
        process.stdout.write(`\r  Writing tensors: ${progressBar(progress.current, progress.total)} ${progress.tensorName?.slice(0, 30) || ''}`.padEnd(80));
      } else if (progress.stage === 'complete') {
        console.log(`\n  Complete!`);
      }
    },
  });

  return result;
}

async function convertSafetensors(inputPath: string, outputDir: string, options: ConvertOptions): Promise<ConvertResult> {
  console.log(`\nParsing Safetensors: ${inputPath}`);

  const parsed = await parseSafetensors(inputPath);
  console.log(`  Tensors: ${parsed.tensors.length}`);

  const isIndex = 'shards' in parsed;
  if (isIndex) {
    const indexParsed = parsed as ParsedSafetensorsIndex;
    console.log(`  Shards: ${indexParsed.shards.length}`);
  }

  const config = parsed.config as ModelConfig | undefined;
  if (config) {
    console.log(`  Architecture: ${config.architectures?.[0] || 'unknown'}`);
    console.log(`  Hidden size: ${config.hidden_size || 'unknown'}`);
    console.log(`  Layers: ${config.num_hidden_layers || 'unknown'}`);
  }

  let tensors: TensorInfo[] = parsed.tensors.map(t => ({
    name: t.name,
    shape: t.shape,
    dtype: t.dtype,
    offset: t.offset,
    size: t.size,
    filePath: t.filePath,
    shardPath: t.shardPath,
  }));

  const modelInfo: ModelInfo = {
    tensors,
    config: config || {},
    shards: isIndex ? (parsed as ParsedSafetensorsIndex).shards : undefined,
  };

  // Load tokenizer.json if present in the input directory
  const inputDir = (await stat(inputPath)).isDirectory() ? inputPath : dirname(inputPath);
  const tokenizerPath = join(inputDir, 'tokenizer.json');
  try {
    const tokenizerData = await readFile(tokenizerPath, 'utf-8');
    const tokenizerJson = JSON.parse(tokenizerData);
    if (tokenizerJson.model) {
      (modelInfo as { tokenizerJson?: unknown }).tokenizerJson = tokenizerJson;
      console.log(`  Tokenizer: ${tokenizerPath}`);
    }
  } catch {
    // tokenizer.json not found or invalid, will skip
  }

  if (options.textOnly) {
    const originalCount = modelInfo.tensors.length;

    modelInfo.tensors = modelInfo.tensors.filter(t => {
      if (t.name.startsWith('vision_tower.')) return false;
      if (t.name.startsWith('multi_modal_projector.')) return false;
      return true;
    });

    for (const tensor of modelInfo.tensors) {
      if (tensor.name.startsWith('language_model.')) {
        tensor.name = tensor.name.replace('language_model.', '');
      }
    }

    if (modelInfo.config?.text_config) {
      const textConfig = modelInfo.config.text_config;
      modelInfo.config = {
        ...modelInfo.config,
        ...textConfig,
        architectures: ['Gemma3ForCausalLM'],
      };
      delete modelInfo.config.text_config;
      delete modelInfo.config.vision_config;
    }

    console.log(`  Text-only: ${originalCount} -> ${modelInfo.tensors.length} tensors`);
  }

  inferConfigFromTensors(modelInfo);

  // Add Q4K layout to config if quantizing
  const targetQuant = options.quantize || 'f16';
  if (targetQuant === 'q4_k_m' && options.q4kLayout !== 'flat') {
    modelInfo.config = modelInfo.config || {};
    (modelInfo.config as Record<string, unknown>).q4kLayout = options.q4kLayout;
    console.log(`  Q4K layout: ${options.q4kLayout}`);
  }

  const shardBuffers = new Map<string, Buffer>();
  if (options.fast) {
    console.log(`  Pre-loading shard files into memory...`);
    const uniqueShards = new Set<string>();
    for (const tensor of modelInfo.tensors) {
      const path = tensor.shardPath || tensor.filePath;
      if (path) uniqueShards.add(path);
    }
    let totalLoaded = 0;
    for (const shardPath of uniqueShards) {
      const buffer = await readFile(shardPath);
      shardBuffers.set(shardPath, buffer);
      totalLoaded += buffer.length;
      console.log(`    Loaded ${basename(shardPath)} (${formatBytes(buffer.length)})`);
    }
    console.log(`  Total loaded: ${formatBytes(totalLoaded)}`);
  }

  const getTensorData = async (tensor: RDRRTensorInfo): Promise<ArrayBuffer> => {
    // Find full tensor info from our local array
    const localTensor = modelInfo.tensors.find(t => t.name === tensor.name)!;
    let data: ArrayBuffer;
    const filePath = localTensor.shardPath || localTensor.filePath;

    if (options.fast && filePath && shardBuffers.has(filePath)) {
      const buffer = shardBuffers.get(filePath)!;
      data = buffer.buffer.slice(localTensor.offset, localTensor.offset + localTensor.size) as ArrayBuffer;
    } else {
      data = await readTensorData(localTensor as SafetensorsTensor);
    }

    const floatDtypes = ['F32', 'F16', 'BF16'];
    const modulesToNotConvert = modelInfo.config?.quantization_config?.modules_to_not_convert ?? null;
    if (targetQuant === 'q4_k_m' && floatDtypes.includes(localTensor.dtype) && shouldQuantize(tensor.name, tensor.shape, {
      quantizeEmbeddings: options.quantizeEmbeddings,
      modulesToNotConvert,
    })) {
      const f32Data = localTensor.dtype === 'F32'
        ? new Float32Array(data)
        : convertToF32(data, localTensor.dtype);

      // Use appropriate Q4K layout for 2D weight matrices
      let quantized: Uint8Array;
      let newShape = tensor.shape;

      if (tensor.shape.length === 2 && options.q4kLayout !== 'flat') {
        const shape2D = tensor.shape as [number, number];

        if (options.q4kLayout === 'column_wise') {
          // Transpose before quantization for coalesced GPU access
          const result = quantizeToQ4KMColumnWise(f32Data, shape2D);
          quantized = result.quantized;
          newShape = result.transposedShape;  // [K, out] instead of [out, K]

          if (options.verbose) {
            console.log(`  Q4K column-wise: ${tensor.name} ${shape2D} -> ${newShape} (transposed)`);
          }
        } else {
          // Row-wise: compatible with fused kernel
          const result = quantizeToQ4KMRowWise(f32Data, shape2D);
          quantized = result.quantized;

          if (options.verbose) {
            console.log(`  Q4K row-wise: ${tensor.name} ${shape2D} (${result.numBlocks} blocks)`);
          }
        }
      } else {
        // Flat packing for 1D or when explicitly requested
        const result = quantizeToQ4KM(f32Data, tensor.shape);
        quantized = result.quantized;

        if (options.verbose && tensor.shape.length === 2) {
          console.log(`  Q4K flat: ${tensor.name} ${tensor.shape} (legacy, not fused-kernel compatible)`);
        }
      }

      localTensor.dtype = 'Q4_K_M';
      localTensor.size = quantized.length;
      // Note: shape update for column_wise would need manifest support
      return quantized.buffer as ArrayBuffer;
    }

    return data;
  };

  // Build conversion metadata and runtime optimizations
  const outputQuant = targetQuant.toUpperCase();
  const conversion = buildConversionInfo(inputPath, options, config?.torch_dtype as string || 'bfloat16', outputQuant);
  const optimizations = buildOptimizations(options, outputQuant);

  const result = await writeRDRR(outputDir, modelInfo, getTensorData, {
    modelId: options.modelId || basename(inputPath).replace(/\.(safetensors|json)$/, ''),
    quantization: outputQuant,
    shardSize: options.shardSize * 1024 * 1024,
    fuseGateUp: options.fuseGateUp,
    conversion,
    optimizations,
    onProgress: (progress: ProgressEvent) => {
      if (progress.stage === 'writing') {
        process.stdout.write(`\r  Writing tensors: ${progressBar(progress.current, progress.total)} ${progress.tensorName?.slice(0, 30) || ''}`.padEnd(80));
      } else if (progress.stage === 'complete') {
        console.log(`\n  Complete!`);
      }
    },
  });

  return result;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const options = parseArgs(args);

  if (options.help) {
    printHelp();
    process.exit(0);
  }

  if (options.test) {
    const outputDir = options.output || options.input || './test-model';
    console.log(`Creating test model in: ${outputDir}`);

    const result = await createTestModel(resolve(outputDir));
    console.log(`\nTest model created:`);
    console.log(`  Shards: ${result.shardCount}`);
    console.log(`  Tensors: ${result.tensorCount}`);
    console.log(`  Total size: ${formatBytes(result.totalSize)}`);
    console.log(`  Manifest: ${result.manifestPath}`);
    process.exit(0);
  }

  if (!options.input) {
    console.error('Error: Input file/directory is required');
    console.error('Use --help for usage information');
    process.exit(1);
  }

  const inputPath = resolve(options.input);

  let outputDir: string;
  if (options.output) {
    outputDir = resolve(options.output);
  } else {
    const inputBasename = basename(inputPath);
    const modelName = inputBasename.replace(/\.(gguf|safetensors)$/i, '').replace(/\s+/g, '-').toLowerCase();
    outputDir = join(__dirname, '..', 'models', modelName);
    console.log(`Output directory not specified, using: ${outputDir}`);
  }

  try {
    // Clean output directory if it exists (default behavior to avoid stale files)
    if (options.cleanOutput) {
      try {
        const outputStat = await stat(outputDir);
        if (outputStat.isDirectory()) {
          console.log(`Cleaning existing output directory: ${outputDir}`);
          await rm(outputDir, { recursive: true, force: true });
        }
      } catch {
        // Directory doesn't exist, nothing to clean
      }
    }

    const format = await detectFormat(inputPath);
    console.log(`Detected format: ${format}`);

    let result: ConvertResult;

    if (format === 'gguf') {
      result = await convertGGUF(inputPath, outputDir, options);
    } else if (format === 'safetensors' || format === 'safetensors-dir' || format === 'safetensors-index') {
      result = await convertSafetensors(inputPath, outputDir, options);
    } else {
      throw new Error(`Unsupported format: ${format}`);
    }

    console.log(`\nConversion complete:`);
    console.log(`  Output: ${outputDir}`);
    console.log(`  Shards: ${result.shardCount}`);
    console.log(`  Tensors: ${result.tensorCount}`);
    console.log(`  Total size: ${formatBytes(result.totalSize)}`);
    console.log(`  Manifest: ${result.manifestPath}`);
    console.log(`\nNote: If model was previously loaded in browser, clear OPFS cache:`);
    console.log(`  - Use the "Clear Cache" button in the model selector UI`);
    console.log(`  - Or call: await deleteModel('${options.modelId || basename(outputDir)}')`);
    console.log(`  - Or delete browser profile: rm -rf doppler/.benchmark-cache/`);

  } catch (error) {
    const err = error as Error;
    console.error(`\nError: ${err.message}`);
    if (options.verbose) {
      console.error(err.stack);
    }
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});

export { parseArgs, detectFormat, convertGGUF, convertSafetensors, inferConfigFromTensors, formatBytes };
