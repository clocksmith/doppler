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
  type WriteRDRROptions,
} from './writer.js';
import { quantizeToQ4KM, float32ToFloat16 } from './quantizer.js';
import { shouldQuantize as shouldQuantizeCore, sanitizeModelId } from './core.js';

// Import config-as-code preset detection
import {
  detectPreset,
  resolvePreset,
  DEFAULT_QUANTIZATION_DEFAULTS,
  type ModelType,
  type RawModelConfigSchema,
  type QuantizationInfoSchema,
} from '../config/index.js';
import { log as debugLog } from '../debug/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface ConvertOptions {
  input: string;
  output: string;
  weightQuant: string | null;     // Weight quantization
  embedQuant: string | null;      // Embedding quantization
  headQuant: string | null;       // LM head quantization
  visionQuant: string | null;     // Vision encoder quantization
  audioQuant: string | null;      // Audio encoder quantization
  projectorQuant: string | null;  // Cross-modal projector quantization
  computePrecision: 'f16' | 'f32' | 'auto' | null;  // Runtime compute precision hint
  kernelPlan: string | null;      // JSON string of kernel plan
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
    weightQuant: null,
    embedQuant: null,
    headQuant: null,
    visionQuant: null,
    audioQuant: null,
    projectorQuant: null,
    computePrecision: null,
    kernelPlan: null,
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
      case '--weight-quant':
      case '-w':
        opts.weightQuant = argv[++i] || null;
        break;
      case '--embed-quant':
      case '-e':
        opts.embedQuant = argv[++i] || null;
        break;
      case '--head-quant':
        opts.headQuant = argv[++i] || null;
        break;
      case '--vision-quant':
        opts.visionQuant = argv[++i] || null;
        break;
      case '--audio-quant':
        opts.audioQuant = argv[++i] || null;
        break;
      case '--projector-quant':
        opts.projectorQuant = argv[++i] || null;
        break;
      case '--compute-precision':
        const cp = argv[++i]?.toLowerCase();
        if (cp === 'f16' || cp === 'f32' || cp === 'auto') {
          opts.computePrecision = cp;
        }
        break;
      case '--kernel-plan':
        opts.kernelPlan = argv[++i] || null;
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

/**
 * Normalize quantization tag to canonical short form.
 *
 * DOPPLER naming uses concise storage-only tags:
 * - q4k = Q4_K_M block quantization (the only Q4 we support)
 * - q6k = Q6_K block quantization
 * - q8_0 = Q8_0 quantization
 * - f16/bf16/f32 = Float formats
 * - fp8e4/fp8e5 = Float8 formats
 * - i4/i8 = Integer formats
 */
function normalizeQuantTag(value: string | null | undefined): string {
  if (!value) return 'f16';
  const lower = value.toLowerCase();

  // Q4_K_M variants → q4k (canonical short form)
  if (lower === 'q4_k_m' || lower === 'q4k' || lower === 'q4' || lower === 'q4km') return 'q4k';
  // Q6_K variants → q6k
  if (lower === 'q6_k' || lower === 'q6k' || lower === 'q6') return 'q6k';
  // Q8_0 (keep as-is, common format)
  if (lower === 'q8_0' || lower === 'q8') return 'q8_0';
  // Float formats
  if (lower === 'f16' || lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'bf16' || lower === 'bfloat16') return 'bf16';
  if (lower === 'f32' || lower === 'fp32' || lower === 'float32') return 'f32';
  // Float8 formats
  if (lower === 'fp8e4' || lower === 'fp8e4m3' || lower === 'e4m3') return 'fp8e4';
  if (lower === 'fp8e5' || lower === 'fp8e5m2' || lower === 'e5m2') return 'fp8e5';
  // Integer formats
  if (lower === 'i8' || lower === 'int8') return 'i8';
  if (lower === 'i4' || lower === 'int4') return 'i4';

  return lower;
}

/**
 * Validate that a quantization type is supported for conversion.
 *
 * Currently implemented:
 * - q4k: Q4_K_M block quantization
 * - f16, f32: Float formats (and bf16 auto-converted to f16)
 *
 * Not yet implemented (will error):
 * - q6k, q8_0: Other block quantization
 * - fp8e4, fp8e5: Float8 formats
 * - i4, i8: Integer formats
 */
function validateQuantType(value: string | null, flagName: string): void {
  if (!value) return;
  const normalized = normalizeQuantTag(value);

  // Supported types
  const supported = ['q4k', 'f16', 'bf16', 'f32'];
  if (supported.includes(normalized)) return;

  // Not yet implemented
  const planned = ['q6k', 'q8_0', 'fp8e4', 'fp8e5', 'i4', 'i8'];
  if (planned.includes(normalized)) {
    throw new Error(
      `Quantization type "${normalized}" is not yet implemented.\n` +
      `Supported types: ${supported.join(', ')}\n` +
      `Planned types: ${planned.join(', ')}`
    );
  }

  throw new Error(`Unknown quantization type: "${value}" (flag: ${flagName})`);
}

function resolveManifestQuantization(quantize: string | null, fallback: string): string {
  if (!quantize) return fallback;
  const normalized = normalizeQuantTag(quantize);
  // Return uppercase for manifest field (display format)
  if (normalized === 'q4k') return 'Q4_K_M';
  if (normalized === 'q6k') return 'Q6_K';
  if (normalized === 'q8_0') return 'Q8_0';
  return normalized.toUpperCase();
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

/**
 * Build variant tag for model naming.
 *
 * Format: w{weights}[-e{embeddings}][-h{head}][-v{vision}][-a{audio}][-t{tts}][-p{projector}]
 *
 * Components are only included if they differ from defaults:
 * - embeddings defaults to weights
 * - head defaults to embeddings
 * - multimodal components only included if present
 *
 * Examples:
 * - "wq4k" (weights Q4K, embeddings same)
 * - "wq4k-ef16" (weights Q4K, embeddings F16)
 * - "wq4k-ef16-hf16" (with explicit head)
 * - "wq4k-vf16-pf16" (multimodal with vision + projector)
 */
function buildVariantTag(info: QuantizationInfoSchema): string {
  const weights = info.weights;
  const embeddings = info.embeddings ?? weights;
  const lmHead = info.lmHead ?? embeddings;

  // Start with weights (always present)
  const parts = [`w${weights}`];

  // Add embeddings only if different from weights
  if (embeddings !== weights) {
    parts.push(`e${embeddings}`);
  }

  // Add head only if different from embeddings
  if (lmHead !== embeddings) {
    parts.push(`h${lmHead}`);
  }

  // Multimodal components (only if present)
  if (info.vision) {
    parts.push(`v${info.vision}`);
  }
  if (info.audio) {
    parts.push(`a${info.audio}`);
  }
  if (info.tts) {
    parts.push(`t${info.tts}`);
  }
  if (info.projector) {
    parts.push(`p${info.projector}`);
  }

  return parts.join('-');
}

/**
 * Build quantization info from conversion options.
 *
 * Handles all component quantization with proper defaults:
 * - weights: from --weight-quant or original dtype (WebGPU-safe)
 * - embeddings: from --embed-quant or defaults to original dtype (WebGPU-safe)
 * - lmHead: from --head-quant or defaults to embeddings
 * - vision/audio/tts/projector: from explicit flags only
 *
 * All float formats are normalized to WebGPU-safe types (bf16 → f16).
 */
function buildQuantizationInfo(
  opts: ConvertOptions,
  originalDtype: string,
  embedDtype: string | null,
  lmHeadDtype: string | null,
  hasVision = false,
  hasAudio = false,
  hasProjector = false
): QuantizationInfoSchema {
  // Validate all explicit quantization flags
  validateQuantType(opts.weightQuant, '--weight-quant');
  validateQuantType(opts.embedQuant, '--embed-quant');
  validateQuantType(opts.headQuant, '--head-quant');
  validateQuantType(opts.visionQuant, '--vision-quant');
  validateQuantType(opts.audioQuant, '--audio-quant');
  validateQuantType(opts.projectorQuant, '--projector-quant');

  // WebGPU only supports F16/F32 for floats - BF16 must convert to F16
  // This must be applied to ALL stored dtypes so naming matches storage
  const webgpuSafe = (dtype: string): string => {
    const normalized = normalizeQuantTag(dtype);
    // BF16 not supported in WebGPU, convert to F16
    if (normalized === 'bf16') return 'f16';
    return normalized;
  };

  // Weights: explicit flag > original dtype, always WebGPU-safe
  const weights = webgpuSafe(opts.weightQuant ?? originalDtype);

  // Embeddings: explicit flag > original dtype (WebGPU-safe)
  let embeddings: string;
  if (opts.embedQuant) {
    embeddings = webgpuSafe(opts.embedQuant);
  } else {
    embeddings = webgpuSafe(embedDtype || originalDtype);
  }

  // Head: explicit flag > original head dtype > embeddings
  // Only add explicit lmHead if it differs from embeddings
  let lmHead: string;
  if (opts.headQuant) {
    lmHead = webgpuSafe(opts.headQuant);
  } else if (lmHeadDtype) {
    // Model has explicit lm_head tensor with known dtype
    lmHead = webgpuSafe(lmHeadDtype);
  } else {
    // No explicit lm_head, will use embeddings (tied weights)
    lmHead = embeddings;
  }

  const info: QuantizationInfoSchema = {
    weights,
    embeddings,
    lmHead: lmHead !== embeddings ? lmHead : undefined,
  };

  // Multimodal components (only if present in model and explicitly set)
  if (hasVision && opts.visionQuant) {
    info.vision = normalizeQuantTag(opts.visionQuant);
  } else if (hasVision && !opts.textOnly) {
    info.vision = DEFAULT_QUANTIZATION_DEFAULTS.visionDtype;
  }

  if (hasAudio && opts.audioQuant) {
    info.audio = normalizeQuantTag(opts.audioQuant);
  } else if (hasAudio) {
    info.audio = DEFAULT_QUANTIZATION_DEFAULTS.audioDtype;
  }

  if (hasProjector && opts.projectorQuant) {
    info.projector = normalizeQuantTag(opts.projectorQuant);
  } else if (hasProjector && !opts.textOnly) {
    info.projector = DEFAULT_QUANTIZATION_DEFAULTS.projectorDtype;
  }

  // Runtime hints (not included in variantTag)
  if (opts.computePrecision) {
    info.compute = opts.computePrecision;
  }

  info.variantTag = buildVariantTag(info);
  return info;
}

function resolveModelId(modelId: string | null, baseName: string, variantTag: string | undefined): string {
  if (modelId) return modelId;
  const base = sanitizeModelId(baseName);
  if (!variantTag) return base;
  return base.endsWith(variantTag) ? base : `${base}-${variantTag}`;
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

Quantization Options:
  --weight-quant, -w <type>  Weight quantization: q4k, f16, f32 (default: preserve original)
  --embed-quant, -e <type>   Embedding quantization (default: preserve original)
  --head-quant <type>        LM head quantization (default: same as embeddings)
  --vision-quant <type>      Vision encoder quantization (multimodal models)
  --audio-quant <type>       Audio encoder quantization (speech models)
  --projector-quant <type>   Cross-modal projector quantization

Runtime Plan (stored in manifest, not in filename):
  --compute-precision <p> Compute precision hint: f16, f32, auto
  --kernel-plan <json>    JSON object with kernel plan overrides

General Options:
  --shard-size <mb>     Shard size in MB (default: 64)
  --model-id <id>       Override model ID in manifest
  --text-only           Extract only text model from multimodal
  --fast                Pre-load all shards into memory (faster, more RAM)
  --verbose, -v         Show detailed progress
  --test [path]         Create tiny test fixture at path (default: ./test-model)
  --help, -h            Show this help

Naming Convention:
  Model IDs use format: {name}-w{weights}[-e{embeddings}][-h{head}][-v{vision}]...

  Examples:
    gemma-2b-wq4k           Weights Q4K, embeddings default to weights
    gemma-2b-wq4k-ef16      Weights Q4K, embeddings F16
    llama-8b-wq4k-ef16-hf16 With separate head quantization
    qwen2-vl-7b-wq4k-vf16   Multimodal with vision encoder

  Quantization tokens:
    q4k = Q4_K_M block quant    f16 = Float16       bf16 = BFloat16
    q6k = Q6_K block quant      f32 = Float32       fp8e4/fp8e5 = Float8
    q8_0 = Q8_0 quant           i4/i8 = Integer

Examples:
  # Convert with Q4K weights, keep embeddings as original (bf16/f16)
  npx tsx src/converter/node-converter.ts \\
    ~/.cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots/*/ \\
    models/gemma-2b \\
    -w q4k

  # Explicit embeddings and head quantization
  npx tsx src/converter/node-converter.ts input models/output \\
    -w q4k -e f16 --head-quant f16

  # Multimodal with explicit vision quantization
  npx tsx src/converter/node-converter.ts input models/output \\
    -w q4k --vision-quant f16 --projector-quant f16

  # Set runtime compute precision hint
  npx tsx src/converter/node-converter.ts input models/output \\
    -w q4k --compute-precision f16

  # Multimodal to text-only
  npx tsx src/converter/node-converter.ts \\
    ~/models/gemma-3-4b-it \\
    models/gemma-4b-text \\
    --text-only -w q4k

  # Create tiny test fixture
  npx tsx src/converter/node-converter.ts --test ./test-model
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
  quantizationInfo?: QuantizationInfoSchema;
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

  // Detect multimodal components
  const visionPatterns = ['vision_', 'vision_tower', 'vision_model', 'image_encoder'];
  const audioPatterns = ['audio_', 'audio_encoder', 'whisper', 'wav2vec'];
  const projectorPatterns = ['multi_modal_projector', 'mm_projector', 'projector'];

  const hasVision = parsed.tensors.some((t: SafetensorsTensor) =>
    visionPatterns.some(p => t.name.toLowerCase().includes(p)));
  const hasAudio = parsed.tensors.some((t: SafetensorsTensor) =>
    audioPatterns.some(p => t.name.toLowerCase().includes(p)));
  const hasProjector = parsed.tensors.some((t: SafetensorsTensor) =>
    projectorPatterns.some(p => t.name.toLowerCase().includes(p)));

  if (hasVision) verboseLog('Detected vision encoder');
  if (hasAudio) verboseLog('Detected audio encoder');
  if (hasProjector) verboseLog('Detected multimodal projector');

  // Filter text-only if requested
  let tensors = parsed.tensors;
  if (opts.textOnly) {
    const textOnlyPatterns = ['language_model', 'model.', 'lm_head', 'embed_tokens'];
    const excludePatterns = [...visionPatterns, ...audioPatterns, ...projectorPatterns, 'image_newline'];

    tensors = tensors.filter((t: SafetensorsTensor) => {
      const isMultimodal = excludePatterns.some(p => t.name.toLowerCase().includes(p));
      if (isMultimodal) {
        verboseLog(`Skipping multimodal tensor: ${t.name}`);
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

  const embedDtypeRaw = findTensorDtype(validTensors, isEmbeddingTensorName);
  const lmHeadDtypeRaw = findTensorDtype(validTensors, isLmHeadTensorName);
  const quantizationInfo = buildQuantizationInfo(
    opts, originalDtype, embedDtypeRaw, lmHeadDtypeRaw,
    hasVision && !opts.textOnly,
    hasAudio,
    hasProjector && !opts.textOnly
  );
  const baseModelId = typeof configRec._name_or_path === 'string'
    ? configRec._name_or_path
    : basename(inputPath);
  const resolvedModelId = resolveModelId(opts.modelId, baseModelId, quantizationInfo.variantTag);
  const manifestQuantization = resolveManifestQuantization(opts.weightQuant, originalDtype);

  // Helper to determine output dtype for a tensor based on its type
  // WebGPU only supports F16/F32 - BF16 must convert to F16
  const toWebGPUDtype = (dtype: string): string => {
    if (dtype === 'q4k') return 'Q4_K_M';
    if (dtype === 'bf16') return 'F16';  // WebGPU doesn't support BF16
    return dtype.toUpperCase();
  };

  const getOutputDtype = (name: string, shape: number[], originalDtype: string): string => {
    const isEmbedding = isEmbeddingTensorName(name);
    const isHead = isLmHeadTensorName(name);

    if (isEmbedding) {
      return toWebGPUDtype(quantizationInfo.embeddings);
    }
    if (isHead) {
      const headQuant = quantizationInfo.lmHead ?? quantizationInfo.embeddings;
      return toWebGPUDtype(headQuant);
    }
    // Regular weight tensor
    if (shouldQuantizeCore(name, shape)) {
      return toWebGPUDtype(quantizationInfo.weights);
    }
    // Non-quantized tensor - still need WebGPU-safe dtype
    if (originalDtype === 'BF16') return 'F16';
    return originalDtype;
  };

  // Create modelInfo with adjusted dtypes for quantized tensors
  const modelInfo: ModelInfo = {
    modelName: resolvedModelId,
    architecture: arch,
    quantization: manifestQuantization,
    quantizationInfo,
    config,
    tokenizerConfig,
    tokenizerJson: parsed.tokenizerJson,
    tensors: validTensors.map((t: SafetensorsTensor) => ({
      name: t.name,
      shape: t.shape,
      dtype: getOutputDtype(t.name, t.shape, t.dtype),
      size: t.size,
    })),
  };

  // Create tensor data getter
  const tensorMap = new Map(validTensors.map((t: SafetensorsTensor) => [t.name, t]));

  const getTensorData = async (info: TensorInfo): Promise<ArrayBuffer> => {
    const tensor = tensorMap.get(info.name);
    if (!tensor) {
      throw new Error(`Tensor not found: ${info.name}`);
    }

    // Read raw tensor data from safetensors file
    const tensorWithPath = { ...tensor, filePath: tensor.filePath || tensor.shardPath || inputPath };
    const data = await readTensorData(tensorWithPath);

    // Determine target dtype from quantizationInfo
    const targetDtype = getOutputDtype(info.name, info.shape, tensor.dtype);

    // Quantize to Q4_K_M if needed
    if (targetDtype === 'Q4_K_M') {
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

    // Convert to F16 if needed
    if (targetDtype === 'F16' && tensor.dtype !== 'F16') {
      verboseLog(`Converting ${info.name} from ${tensor.dtype} to F16`);

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
      } else if (tensor.dtype === 'F32') {
        f32 = new Float32Array(data);
      } else {
        verboseLog(`Warning: Unknown dtype ${tensor.dtype}, treating as F32`);
        f32 = new Float32Array(data);
      }

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
  const writerOpts: WriteRDRROptions = {
    shardSize: opts.shardSize * 1024 * 1024,
    modelId: resolvedModelId,
    modelType,
    architecture: arch,
    quantization: manifestQuantization,
    quantizationInfo,
  };

  // Add kernel plan if specified
  if (opts.kernelPlan) {
    try {
      writerOpts.optimizations = {
        kernelPlan: JSON.parse(opts.kernelPlan),
      };
    } catch (e) {
      throw new Error(`Invalid --kernel-plan JSON: ${(e as Error).message}`);
    }
  }

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

  const embedDtypeRaw = findTensorDtype(tensors, isEmbeddingTensorName);
  const lmHeadDtypeRaw = findTensorDtype(tensors, isLmHeadTensorName);
  const quantizationInfo = buildQuantizationInfo(
    opts,
    parsed.quantization || 'F32',
    embedDtypeRaw,
    lmHeadDtypeRaw
  );
  const resolvedModelId = resolveModelId(opts.modelId, modelName, quantizationInfo.variantTag);
  const manifestQuantization = resolveManifestQuantization(opts.weightQuant, parsed.quantization || 'F32');

  const modelInfo: ModelInfo = {
    modelName: resolvedModelId,
    architecture: arch,
    quantization: manifestQuantization,
    quantizationInfo,
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
    const writerOpts: WriteRDRROptions = {
      shardSize: opts.shardSize * 1024 * 1024,
      modelId: resolvedModelId,
      modelType,
      architecture: arch,
      quantization: manifestQuantization,
      quantizationInfo,
    };

    // Add kernel plan if specified
    if (opts.kernelPlan) {
      try {
        writerOpts.optimizations = {
          kernelPlan: JSON.parse(opts.kernelPlan),
        };
      } catch (e) {
        throw new Error(`Invalid --kernel-plan JSON: ${(e as Error).message}`);
      }
    }

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
  if (opts.weightQuant) {
    console.log(`  Weight quantization: ${opts.weightQuant}`);
  }
  if (opts.embedQuant) {
    console.log(`  Embed quantization: ${opts.embedQuant}`);
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
