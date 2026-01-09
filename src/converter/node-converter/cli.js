/**
 * Command-line argument parsing for the Node.js Model Converter.
 *
 * Handles parsing of CLI arguments and help text generation.
 *
 * @module converter/node-converter/cli
 */

import { createConverterConfig } from '../../config/schema/index.js';

const DEFAULT_CONFIG = createConverterConfig();
const BYTES_PER_MB = 1024 * 1024;
const DEFAULT_SHARD_SIZE_MB = Math.round(
  DEFAULT_CONFIG.sharding.shardSizeBytes / BYTES_PER_MB
);

/**
 * Parse command-line arguments into ConvertOptions.
 */
export function parseArgs(argv) {
  const opts = {
    input: '',
    output: '',
    weightQuant: DEFAULT_CONFIG.quantization.weights,
    embedQuant: DEFAULT_CONFIG.quantization.embeddings,
    headQuant: DEFAULT_CONFIG.quantization.lmHead,
    visionQuant: DEFAULT_CONFIG.quantization.vision,
    audioQuant: DEFAULT_CONFIG.quantization.audio,
    projectorQuant: DEFAULT_CONFIG.quantization.projector,
    computePrecision: DEFAULT_CONFIG.quantization.computePrecision,
    shardSize: DEFAULT_SHARD_SIZE_MB,
    modelId: DEFAULT_CONFIG.output.modelId,
    textOnly: DEFAULT_CONFIG.output.textOnly,
    fast: DEFAULT_CONFIG.output.fast,
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
      case '--compute-precision': {
        const cp = argv[++i]?.toLowerCase();
        if (cp === 'f16' || cp === 'f32' || cp === 'auto') {
          opts.computePrecision = cp;
        }
        break;
      }
      case '--shard-size': {
        const value = argv[++i];
        opts.shardSize = value ? parseInt(value, 10) : DEFAULT_SHARD_SIZE_MB;
        break;
      }
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
 * Print help text to console.
 */
export function printHelp() {
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

General Options:
  --shard-size <mb>     Shard size in MB (default: ${DEFAULT_SHARD_SIZE_MB})
  --model-id <id>       Base model ID (variant tag auto-appended)
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
