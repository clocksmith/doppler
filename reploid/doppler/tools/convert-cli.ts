#!/usr/bin/env node
/**
 * Model Conversion CLI - Convert HuggingFace/GGUF models to RDRR format.
 *
 * Usage:
 *   npx tsx tools/convert-cli.ts <input> <output> [options]
 *
 * Examples:
 *   npx tsx tools/convert-cli.ts ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/  models/gemma-1b
 *   npx tsx tools/convert-cli.ts model.gguf models/my-model --quantize q4_k_m
 *   npx tsx tools/convert-cli.ts --test ./test-model  # Create tiny test fixture
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
} from './safetensors-parser.js';
import {
  parseGGUF,
  parseGGUFFile,
  getTensors,
} from './gguf-parser.js';
import {
  writeRDRR,
  createTestModel,
  type WriteResult,
  type WriterOptions,
} from './rdrr-writer.js';
import { quantizeToQ4KM, float32ToFloat16 } from './quantizer.js';

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

function printHelp(): void {
  console.log(`
Model Conversion CLI - Convert HuggingFace/GGUF models to RDRR format.

Usage:
  npx tsx tools/convert-cli.ts <input> <output> [options]
  npx tsx tools/convert-cli.ts --test [output]

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
  npx tsx tools/convert-cli.ts \\
    ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*/ \\
    models/gemma-1b \\
    --quantize q4_k_m

  # Convert GGUF file
  npx tsx tools/convert-cli.ts model.gguf models/my-model

  # Multimodal to text-only
  npx tsx tools/convert-cli.ts \\
    ~/models/gemma-3-4b-it \\
    models/gemma-4b-text \\
    --text-only --quantize q4_k_m

  # Create tiny test fixture
  npx tsx tools/convert-cli.ts --test ./test-model
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
  const log = (msg: string) => opts.verbose && console.log(`[Convert] ${msg}`);

  log(`Parsing safetensors from: ${inputPath}`);
  const parsed = await parseSafetensors(inputPath);
  log(`Found ${parsed.tensors.length} tensors`);

  // Load config and tokenizer
  const config = await loadModelConfig(inputPath).catch(() => ({}));
  const tokenizerConfig = await loadTokenizerConfig(inputPath).catch(() => undefined);

  // Detect architecture
  const arch = (config.architectures as string[])?.[0] ||
    (config as Record<string, string>).model_type ||
    'llama';

  log(`Architecture: ${arch}`);
  log(`Config: ${JSON.stringify(config, null, 2).slice(0, 500)}...`);

  // Filter text-only if requested
  let tensors = parsed.tensors;
  if (opts.textOnly) {
    const textOnlyPatterns = ['language_model', 'model.', 'lm_head', 'embed_tokens'];
    const visionPatterns = ['vision_', 'vision_tower', 'multi_modal_projector', 'image_newline'];

    tensors = tensors.filter((t: SafetensorsTensor) => {
      const isVision = visionPatterns.some(p => t.name.includes(p));
      if (isVision) {
        log(`Skipping vision tensor: ${t.name}`);
        return false;
      }
      return true;
    });
    log(`Filtered to ${tensors.length} text-only tensors`);
  }

  // Create modelInfo
  const modelInfo: ModelInfo = {
    modelName: opts.modelId || basename(inputPath),
    architecture: arch,
    quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : 'F32',
    config,
    tokenizerConfig,
    tokenizerJson: parsed.tokenizerJson,
    tensors: tensors.map((t: SafetensorsTensor) => ({
      name: t.name,
      shape: t.shape,
      dtype: t.dtype,
    })),
  };

  // Create tensor data getter
  const tensorMap = new Map(tensors.map((t: SafetensorsTensor) => [t.name, t]));

  const getTensorData = async (info: TensorInfo): Promise<ArrayBuffer> => {
    const tensor = tensorMap.get(info.name);
    if (!tensor) {
      throw new Error(`Tensor not found: ${info.name}`);
    }

    // Read raw tensor data from safetensors file
    const data = await readTensorData(tensor.filePath || inputPath, tensor);

    // Quantize if requested
    if (opts.quantize === 'q4_k_m') {
      const isEmbedding = info.name.includes('embed') || info.name.includes('lm_head');
      if (!isEmbedding || opts.quantizeEmbeddings) {
        log(`Quantizing ${info.name} to Q4_K_M`);
        const f32 = new Float32Array(data);
        const q4 = await quantizeToQ4KM(f32);
        return q4.buffer;
      }
    }

    if (opts.quantize === 'f16' && tensor.dtype !== 'F16') {
      log(`Converting ${info.name} to F16`);
      const f32 = new Float32Array(data);
      const f16 = float32ToFloat16(f32);
      return f16.buffer;
    }

    return data;
  };

  // Write RDRR
  log(`Writing RDRR to: ${outputPath}`);
  const writerOpts: WriterOptions = {
    shardSize: opts.shardSize * 1024 * 1024,
    modelId: opts.modelId || basename(inputPath),
    architecture: arch,
    quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : 'F32',
  };

  return writeRDRR(outputPath, modelInfo, getTensorData, writerOpts);
}

async function convertGGUF(
  inputPath: string,
  outputPath: string,
  opts: ConvertOptions
): Promise<WriteResult> {
  const log = (msg: string) => opts.verbose && console.log(`[Convert] ${msg}`);

  log(`Parsing GGUF from: ${inputPath}`);
  const parsed = await parseGGUFFile(inputPath);
  log(`Found ${parsed.tensors.length} tensors`);

  // Extract model config from GGUF metadata
  const metadata = parsed.config;
  const arch = (metadata.general_architecture as string) || 'llama';
  const modelName = (metadata.general_name as string) || basename(inputPath, '.gguf');

  const config: Record<string, unknown> = {
    architectures: [arch],
    model_type: arch,
    hidden_size: metadata[`${arch}.embedding_length`] || metadata.embedding_length,
    num_hidden_layers: metadata[`${arch}.block_count`] || metadata.block_count,
    num_attention_heads: metadata[`${arch}.attention.head_count`] || metadata.head_count,
    num_key_value_heads: metadata[`${arch}.attention.head_count_kv`] || metadata.head_count_kv,
    vocab_size: metadata[`${arch}.vocab_size`] || metadata.vocab_size,
    intermediate_size: metadata[`${arch}.feed_forward_length`] || metadata.feed_forward_length,
    rms_norm_eps: metadata[`${arch}.attention.layer_norm_rms_epsilon`] || 1e-5,
    rope_theta: metadata[`${arch}.rope.freq_base`] || 10000,
    max_position_embeddings: metadata[`${arch}.context_length`] || 4096,
  };

  log(`Architecture: ${arch}`);
  log(`Model: ${modelName}`);

  const tensors = parsed.tensors;
  const modelInfo: ModelInfo = {
    modelName: opts.modelId || modelName,
    architecture: arch,
    quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : parsed.quantization || 'F32',
    config,
    tensors: tensors.map(t => ({
      name: t.name,
      shape: t.shape,
      dtype: t.dtype,
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
    log(`Writing RDRR to: ${outputPath}`);
    const writerOpts: WriterOptions = {
      shardSize: opts.shardSize * 1024 * 1024,
      modelId: opts.modelId || modelName,
      architecture: arch,
      quantization: opts.quantize === 'q4_k_m' ? 'Q4_K_M' : opts.quantize === 'f16' ? 'F16' : parsed.quantization || 'F32',
    };

    return writeRDRR(outputPath, modelInfo, getTensorData, writerOpts);
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
